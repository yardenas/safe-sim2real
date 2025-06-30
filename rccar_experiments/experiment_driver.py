import threading
from collections import defaultdict

import numpy as np

from rccar_experiments.collect_trajectory import TrajectoryCollector
from rccar_experiments.session import Session
from rccar_experiments.transitions_server import TransitionsServer


class ExperimentDriver(Node):
    def __init__(self):
        super().__init__("joystick_publisher")
        self.declare_parameter("cmd_a", [0.65, 0.35, 1.3])
        self.declare_parameter("cmd_b", [0.99, 0.99, 0.99])
        self.declare_parameter("trajectory_length", 1000)
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("seed", 42)
        self.declare_parameter("limit_scale_factor", 0.75)
        self.declare_parameter("joint_limit_budget", 100)
        self.declare_parameter("session_id", "session_0")
        self.joint_limit_budget = self.get_parameter("joint_limit_budget")
        self.dt = self.get_parameter("dt").value
        session_id = self.get_parameter("session_id").value
        self.session = Session(filename=session_id, directory="experiment_sessions")
        num_steps = len(self.session.steps)
        if self.session.steps == 0:
            seed = self.get_parameter("seed").value
        else:
            seed = num_steps
        self.command_sampler = CommandSampler(
            np.asarray(self.get_parameter("cmd_a").value),
            np.asarray(self.get_parameter("cmd_b").value),
            self.dt,
            seed,
        )
        self.trajectory_length = self.get_parameter("trajectory_length").value
        self.trajectory_collector = TrajectoryCollector(
            self.get_parameter("limit_scale_factor").value,
            self.get_parameter("joint_limit_budget").value,
            self.trajectory_length,
        )
        self.command_publisher = self.create_publisher(Remote, "monitor_joystick", 10)
        self.joystick_subscriber = self.create_subscription(
            Remote, "unitree_joystick", self.joystick_callback, 10
        )
        self.transitions_subscription = self.create_subscription(
            Transition, "transition", self.trajectory_collector.transitions_callback, 10
        )
        self.monitor_subscription = self.create_subscription(
            Monitor, "monitor", self.trajectory_collector.monitor_callback, 10
        )
        self.fsm_subscription = self.create_subscription(
            StateInfo,
            "FSM_node_robot_ONBOARD/state_info",
            self.fsm_callback,
            10,
        )
        self.start_service = self.create_service(
            Trigger, "start_sampling", self.start_sampling_callback
        )
        self.cli = self.create_client(SetParameters, "controller/set_parameters")
        self.running = False
        self.run_id = num_steps
        self.timer = self.create_timer(self.dt, self.timer_callback)
        self.timer.cancel()  # Start with the timer off
        self.transitions_server = TransitionsServer(self)
        # Create and start the loop thread
        self.server_thread = threading.Thread(
            target=self.transitions_server.loop, daemon=True
        )
        self.server_thread.start()
        self.get_logger().info("Experiment driver initialized.")
        self.fsm_state = 0
        self.joystick_command = np.zeros(3)

    def joystick_callback(self, msg):
        self.joystick_command = np.array(
            (msg.speed_forward, msg.speed_sideways, msg.speed_turning)
        )
        if not self.running and np.linalg.norm(self.joystick_command) > 0.05:
            self.command_publisher.publish(_populate_command(self.joystick_command))

    def start_sampling_callback(self, request, response):
        del request
        if self.running:
            response.success = False
            response.message = "Sampling is already running."
            return response
        self.get_logger().info(f"Starting command sampling... Run id: {self.run_id}")
        if not self.trajectory_collector.is_standing:
            self.get_logger().info("Cannot start sampling. Robot is not standing.")
            response.success = False
            response.message = "Robot is not standing."
            return response
        self.running = True
        self.command_sampler.start()
        self.trajectory_collector.start()
        self.timer.reset()
        response.success = True
        response.message = "Command sampling started."
        return response

    def timer_callback(self):
        if not self.running:
            return
        if self.trajectory_collector.trajectory_done:
            self.get_logger().info(
                "Command sampling completed. Returning to standing mode."
            )
            self.running = False
            self.timer.cancel()
            self.command_sampler.end()
            self.command_publisher.publish(
                _populate_command(self.command_sampler.x_k, stand=True)
            )
            self.run_id += 1
            self.summarize_trial()
            self.trajectory_collector.end()
            return
        if np.linalg.norm(self.joystick_command) > 0.05:
            command = self.joystick_command
        else:
            command = self.command_sampler.step()
        self.command_publisher.publish(_populate_command(command))

    def summarize_trial(self):
        infos = [
            transition.info for transition in self.trajectory_collector.transitions
        ]
        table_data = defaultdict(float)
        for info in infos:
            for key, value in info.items():
                table_data[key] += value
        table_data["steps"] = len(infos)
        table_data["reward"] = self.trajectory_collector.reward
        table_data["cost"] = self.trajectory_collector.joint_limit_steps
        self.get_logger().info(
            f"Total reward: {self.trajectory_collector.reward}\nTotal cost: {self.trajectory_collector.joint_limit_steps}\n{_format_reward_summary(table_data)}"
        )
        self.session.update(table_data)

    def update_policy(self, new_path):
        self.get_logger().info("Updating policy...")
        # Create the parameter value
        param = Parameter()
        param.name = "policy_path"
        param.value = ParameterValue()
        param.value.string_value = new_path
        param.value.type = ParameterType.PARAMETER_STRING
        # Create the request
        request = SetParameters.Request()
        request.parameters = [param]
        # Call the service
        response = self.cli.call(request)
        self.get_logger().info("Parameter update result: %s" % str(response))
        if response.results[0].successful:
            return True
        else:
            self.get_logger().error("Failed to update parameter.")
            return False

    def get_trajectory(self):
        return self.trajectory_collector.transitions

    @property
    def robot_ok(self):
        return self.trajectory_collector.is_standing and self.fsm_state == 2

    def fsm_callback(self, msg):
        # Transition from walking to estop terminates
        if self.fsm_state != 2 and msg.cur_state == 3:
            self.trajectory_collector.set_terminated = True
        self.fsm_state = msg.cur_state


def _format_reward_summary(table_data):
    lines = []
    header = f"{'Reward Component':<20} {'Total Value':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for key, value in table_data.items():
        lines.append(f"{key:<20} {value:>12.2f}")
    return "\n".join(lines)


def _populate_command(command, *, stand=False):
    msg = Remote()
    if stand:
        msg.gait = 0
    else:
        msg.gait = 1
    msg.speed_forward = float(command[0])
    msg.speed_sideways = float(command[1])
    msg.speed_turning = float(command[2])
    return msg
