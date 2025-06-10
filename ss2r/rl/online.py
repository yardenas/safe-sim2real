import functools
from typing import Sequence, Tuple

import cloudpickle as pickle
import jax
import jax.numpy as jnp
import zmq
from brax import envs
from brax.training import acting
from brax.training.types import PolicyParams, PRNGKey, Transition
from jax.experimental import io_callback

from ss2r.rl.types import MakePolicyFn


class OnlineEpisodeOrchestrator:
    def __init__(
        self,
        translate_policy_to_binary_fn,
        data_postprocess_fn,
        num_steps,
        address="tcp://localhost:5555",
    ):
        """Orchestrator for requesting episodes over ZMQ, with optional SSH reverse tunnel.

        If open_reverse_tunnel=True, a reverse tunnel will be opened from the remote machine
        (client) back to this machine (server) using SSH:
            ssh -R remote_tunnel_port:localhost:local_zmq_port ssh_server

        The remote client will connect to tcp://localhost:<remote_tunnel_port>
        which routes to this machine's local ZMQ server.

        Parameters
        ----------
        translate_policy_to_binary_fn : function
            Converts policies to a serializable format.
        address : str
            ZMQ address to bind or connect to (default: tcp://localhost:5555).
        open_reverse_tunnel : bool
            Whether to open an SSH reverse tunnel to expose this address remotely.
        ssh_server : str
            SSH target (e.g., 'user@host[:port]') to tunnel through.
        """
        self._translate_policy_to_binary_fn = translate_policy_to_binary_fn
        self._data_postprocess_fn = data_postprocess_fn
        self.num_steps = num_steps
        self._address = address

    def request_data(
        self,
        env: envs.Env,
        env_state: envs.State,
        make_policy_fn: MakePolicyFn,
        policy_params: PolicyParams,
        key: PRNGKey,
        *,
        extra_fields: Sequence[str],
    ) -> Tuple[envs.State, Transition]:
        dummy_transitions = acting.actor_step(
            env,
            env_state,
            make_policy_fn(policy_params),
            key,
            extra_fields,
        )[1]
        dummy_transitions = jax.tree.map(
            lambda x: jnp.tile(x, (self.num_steps,) + (1,) * (x.ndim - 1)),
            dummy_transitions,
        )
        transitions = io_callback(
            functools.partial(self._send_request, make_policy_fn, extra_fields),
            dummy_transitions,
            policy_params,
            ordered=True,
        )
        return env_state, transitions

    def _send_request(self, make_policy_fn, extra_fields, policy_params):
        policy_bytes = self._translate_policy_to_binary_fn(
            make_policy_fn, policy_params
        )
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REQ) as socket:
                socket.connect(self._address)
                while True:
                    print("Requesting data...")
                    # Send data
                    socket.send(pickle.dumps((policy_bytes, self.num_steps)))
                    # Receive response
                    raw_data = pickle.loads(socket.recv())
                    transitions = self._data_postprocess_fn(raw_data, extra_fields)
                    print(f"Received {len(transitions.reward)} transitions...")
                    return transitions
