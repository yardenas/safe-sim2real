import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.core import Wrapper
from gymnasium.spaces import Box
from PIL import Image


class ActionRepeat(Wrapper):
    def __init__(self, env, repeat):
        assert repeat >= 1, "Expects at least one repeat."
        super(ActionRepeat, self).__init__(env)
        self.repeat = repeat

    def step(self, action):
        done = False
        total_reward = 0.0
        total_cost = 0.0
        current_step = 0
        info = {"steps": 0}
        while current_step < self.repeat and not done:
            obs, reward, terminal, truncated, info = self.env.step(action)
            total_reward += reward
            total_cost += info.get("cost", 0.0)
            current_step += 1
            done = truncated or terminal
        info["steps"] = current_step
        info["cost"] = total_cost
        return obs, total_reward, terminal, truncated, info


class ImageObservation(ObservationWrapper):
    def __init__(
        self, env, image_size, image_format="channels_first", *, render_kwargs=None
    ):
        super(ImageObservation, self).__init__(env)
        assert image_format in ["channels_first", "channels_last"]
        size = image_size + (3,) if image_format == "chw" else (3,) + image_size
        self.observation_space = Box(0, 255, size, np.float32)
        if render_kwargs is None:
            render_kwargs = {}
        self._render_kwargs = render_kwargs
        self.image_size = image_size
        self.image_format = image_format

    def observation(self, observation):
        image = self.env.render(**self._render_kwargs)
        image = Image.fromarray(image)
        if image.size != self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image, copy=False)
        if self.image_format == "channels_first":
            image = np.moveaxis(image, -1, 0)
        image = np.clip(image, 0, 255).astype(np.float32)
        return image


class ChannelFirst(ObservationWrapper):
    def __init__(self, env):
        super(ChannelFirst, self).__init__(env)
        shape = self.unwrapped.observation_space.shape
        assert isinstance(shape, tuple) and len(shape) == 3
        self.observation_space = Box(0, 255, (shape[2], shape[0], shape[1]), np.float32)

    def observation(self, observation):
        image = np.moveaxis(observation, -1, 0)
        return image


class SauteMDP(Wrapper):
    def __init__(
        self, env, safety_budget: float, discount_factor: float, penalty: float
    ):
        super(SauteMDP, self).__init__(env)
        self.safety_budget = safety_budget
        self.discount_factor = discount_factor
        self.penalty = penalty
        assert isinstance(env.observation_space, Box)
        low = np.concatenate([env.observation_space.low, [-np.inf]])
        high = np.concatenate([env.observation_space.high, [safety_budget]])
        self.observation_space = Box(
            low, high, low.shape, env.observation_space.dtype
        )
        self.state = self.safety_budget

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        cost = info.get("cost", 0.0)
        reward = reward if self.state >= 0.0 else -self.penalty
        self.state = (self.state - cost) / self.discount_factor
        obs = self._observation(obs)
        return obs, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        self.state = self.safety_budget
        obs, info = super().reset(seed=seed, options=options)
        return self._observation(obs), info

    def _observation(self, obs):
        return np.concatenate([obs, [self.state]])
