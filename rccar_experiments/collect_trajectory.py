class TrajectoryCollector:
    def __init__(
        self,
        trajectory_length=250,
    ):
        self.current_step = 0
        self.trajectory_length = trajectory_length
        self.reward = 0
        self._monitor_messages_received = 0
        self.transitions = []
        self.running = False
        self.is_standing = False
        self.terminated = False
        self.set_terminated = False

    def transitions_callback(self, msg):
        if not self.running:
            self.is_standing = not msg.done
            return
        self.current_step += 1
        self.terminated = msg.done or self.set_terminated
        truncation = self.current_step >= self.trajectory_length and not self.terminated
        transition = _make_transition(msg, self.terminated, truncation)
        self.reward += transition.reward
        self.transitions.append(transition)

    def start(self):
        self.current_step = 0
        self.joint_limit_counter = 0
        self.reward = 0
        self.transitions = []
        self._monitor_messages_received = 0
        self.terminated = False
        self.set_terminated = False
        self.running = True

    def end(self):
        self.running = False

    @property
    def trajectory_done(self):
        return self.current_step >= self.trajectory_length or self.terminated


def _make_transition(msg, terminated, truncated):
    observation = {
        "state": _make_state(msg.observation),
        "privileged_state": _make_privileged_state(msg.observation),
    }
    next_observation = {
        "state": _make_state(msg.next_observation),
        "privileged_state": _make_privileged_state(msg.next_observation),
    }
    info = {kv.key: kv.value for kv in msg.info}
    reward = msg.reward
    # Correct reward for estops
    if terminated and not msg.done:
        reward -= 1
    info["truncation"] = truncated
    info["termination"] = -terminated
    return Transition(
        observation=observation,
        action=msg.action,
        reward=reward,
        next_observation=next_observation,
        done=msg.done,
        info=info,
    )
