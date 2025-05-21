import atexit
import functools
import os
import re
from getpass import getpass
from typing import Sequence, Tuple

import cloudpickle as pickle
import pexpect
import zmq
from brax import envs
from brax.training import acting
from brax.training.types import PolicyParams, PRNGKey
from jax.experimental import io_callback

from ss2r.algorithms.sac import MakePolicyFn, Transition


class OnlineEpisodeOrchestrator:
    def __init__(
        self,
        translate_policy_to_binary_fn,
        num_steps,
        address="tcp://localhost:5555",
        open_reverse_tunnel=False,
        ssh_server=None,
        local_zmq_port=5559,
        remote_tunnel_port=5555,
        ssh_timeout=60,
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
        self.num_steps = num_steps
        self._address = address
        self._tunnel_pid = None
        if open_reverse_tunnel:
            if ssh_server is None:
                raise ValueError(
                    "ssh_server must be provided if open_reverse_tunnel=True"
                )
            # Launch the tunnel
            self._tunnel_pid = openssh_reverse_tunnel(
                rport=remote_tunnel_port,
                lport=local_zmq_port,
                server=ssh_server,
                localip="127.0.0.1",
                timeout=ssh_timeout,
            )
            print(
                f"[ZMQ Tunnel] Reverse SSH tunnel started with PID {self._tunnel_pid}"
            )

            def _cleanup_tunnel():
                if self._tunnel_pid:
                    try:
                        print(
                            f"[ZMQ Tunnel] Cleaning up SSH tunnel PID {self._tunnel_pid}"
                        )
                        os.kill(self._tunnel_pid, 15)  # send SIGTERM
                    except Exception as e:
                        print(f"[ZMQ Tunnel] Failed to clean up tunnel: {e}")

            atexit.register(_cleanup_tunnel)

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
        dummy_transitions = acting.generate_unroll(
            env,
            env_state,
            make_policy_fn(policy_params),
            key,
            self.num_steps,
            extra_fields,
        )[1]
        transitions = io_callback(
            functools.partial(self._send_request, make_policy_fn),
            dummy_transitions,
            policy_params,
            ordered=True,
        )
        return env_state, transitions

    def _send_request(self, make_policy_fn, policy_params):
        print("Requesting more data...")
        # policy_bytes = self._translate_policy_to_binary_fn(
        #     make_policy_fn, policy_params
        # )
        policy_bytes = pickle.dumps(policy_params)
        with zmq.Context() as ctx:
            with ctx.socket(zmq.REQ) as socket:
                socket.connect(self._address)
                # Send data
                socket.send(pickle.dumps((policy_bytes, self.num_steps)))
                # Receive response
                response = pickle.loads(socket.recv())
                return response


_password_pat = re.compile(rb"pass(word|phrase)", re.IGNORECASE)


class SSHException(Exception):  # type: ignore
    pass


class MaxRetryExceeded(Exception):
    pass


# FIXME (yarden): this actually does not happen here, but on the server host.
# Adapted from https://github.com/zeromq/pyzmq/blob/a37d49db40f1836e3f72297b136b87f0b05ab2e2/zmq/ssh/tunnel.py#L206
def openssh_reverse_tunnel(
    rport, lport, server, localip="127.0.0.1", keyfile=None, password=None, timeout=60
):
    """Create a reverse ssh tunnel using command-line ssh that connects port `rport`
    on the *remote* server to `localhost:lport` on this machine.

    Equivalent to: ssh -R rport:localhost:lport user@server

    Parameters
    ----------
    rport : int
        Port on the remote machine that will forward to this machine.
    lport : int
        Local port on this machine to receive forwarded connections.
    server : str
        SSH target, e.g. 'user@hostname[:port]'.
    localip : str
        Local IP to use for forwarding (default: 127.0.0.1).
    keyfile : str
        Path to private key (optional).
    password : str
        Password for SSH login (if no key).
    timeout : int
        Seconds to keep the tunnel open.
    """
    ssh = "ssh "
    if keyfile:
        ssh += "-i " + keyfile

    if ":" in server:
        server, port = server.split(":")
        ssh += f" -p {port}"
    cmd = f"{ssh} -f -S none -R {rport}:{localip}:{lport} {server} sleep {timeout}"
    # pop SSH_ASKPASS from env
    env = os.environ.copy()
    env.pop("SSH_ASKPASS", None)

    ssh_newkey = "Are you sure you want to continue connecting"
    tunnel = pexpect.spawn(cmd, env=env)
    failed = False
    MAX_RETRY = 10
    for _ in range(MAX_RETRY):
        try:
            i = tunnel.expect([ssh_newkey, _password_pat], timeout=0.1)
            if i == 0:
                raise SSHException("The authenticity of the host can't be established.")
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            if tunnel.exitstatus:
                print(tunnel.exitstatus)
                print(tunnel.before)
                print(tunnel.after)
                raise RuntimeError(f"tunnel '{cmd}' failed to start")
            else:
                return tunnel.pid
        else:
            if failed:
                print("Password rejected, try again")
                password = None
            if password is None:
                password = getpass(f"{server}'s password: ")
            tunnel.sendline(password)
            failed = True
    raise MaxRetryExceeded(f"Failed after {MAX_RETRY} attempts")
