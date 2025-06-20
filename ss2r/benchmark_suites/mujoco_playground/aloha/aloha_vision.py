# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distillation module for sim-to-real
transfer of ALOHA peg insertion."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import manipulation
from mujoco_playground._src.manipulation.aloha import (
    depth_noise,
    distillation,
    peg_insertion,
    pick_base,
)

domain_randomize = distillation.domain_randomize


class PegInsertionVision(peg_insertion.SinglePegInsertion):
    """Distillation environment for peg insertion task with vision capabilities.

    This class extends the PegInsertion environment to support policy distillation
    with vision-based observations, including depth and RGB camera inputs.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = distillation.default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides, distill=True)
        self._vision = config.vision
        if self._vision:
            # Import here to avoid dependency issues when vision is disabled
            # pylint: disable=import-outside-toplevel
            from madrona_mjx.renderer import BatchRenderer

            render_height = self._config.vision_config.render_height
            render_width = self._config.vision_config.render_width
            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=self._config.vision_config.gpu_id,
                num_worlds=self._config.vision_config.render_batch_size,
                batch_render_view_height=render_height,
                batch_render_view_width=render_width,
                enabled_geom_groups=np.asarray(
                    self._config.vision_config.enabled_geom_groups
                ),
                enabled_cameras=np.asarray(self._config.vision_config.enabled_cameras),
                add_cam_debug_geo=False,
                use_rasterizer=self._config.vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )
            self.max_depth = {"pixels/view_0": 0.4, "pixels/view_1": 0.4}

            if self._config.obs_noise.depth:
                # color range based on max_depth values.
                # Pre-sample random lines for simplicity.
                max_depth = self.max_depth["pixels/view_0"]
                self.line_bank = jp.array(
                    depth_noise.np_get_line_bank(
                        render_height,
                        render_width,
                        bank_size=16384,
                        color_range=[max_depth * 0.2, max_depth * 0.85],
                    )
                )

    def reset_color_noise(self, info):
        info["rng"], rng_brightness = jax.random.split(info["rng"])

        info["brightness"] = jax.random.uniform(
            rng_brightness,
            (),
            minval=self._config.obs_noise.brightness[0],
            maxval=self._config.obs_noise.brightness[1],
        )

        info["color_noise"] = {}
        info["shade_noise"] = {}  # Darkness of the colored object.

        color_noise_scales = {0: 0.3, 2: 0.05}
        shade_noise_mins = {0: 0.5, 2: 0.9}
        shade_noise_maxes = {0: 1.0, 2: 2.0}

        def generate_noise(chan):
            info["rng"], rng_noise, rng_shade = jax.random.split(info["rng"], 3)
            noise = jax.random.uniform(
                rng_noise, (1, 3), minval=0, maxval=color_noise_scales[chan]
            )
            noise = noise.at[0, chan].set(0)
            info["color_noise"][chan] = noise
            info["shade_noise"][chan] = jax.random.uniform(
                rng_shade,
                (),
                minval=shade_noise_mins[chan],
                maxval=shade_noise_maxes[chan],
            )

        for chan in [0, 2]:
            generate_noise(chan)

    def _get_obs_distill(self, data, info, init=False):
        obs_pick = self._get_obs_pick(data, info)
        obs_insertion = jp.concatenate([obs_pick, self._get_obs_dist(data, info)])
        if not self._vision:
            state_wt = jp.concatenate(
                [
                    obs_insertion,
                    (info["_steps"] / self._config.episode_length).reshape(1),
                ]
            )
            return {"state_with_time": state_wt}
        if init:
            info["render_token"], rgb, depth = self.renderer.init(data, self._mjx_model)
        else:
            _, rgb, depth = self.renderer.render(info["render_token"], data)
        # Process depth.
        info["rng"], rng_l, rng_r = jax.random.split(info["rng"], 3)
        dmap_l = self.process_depth(depth, 0, "pixels/view_0", rng_l)
        r_dmap_l = jax.image.resize(dmap_l, (8, 8, 1), method="nearest")
        dmap_r = self.process_depth(depth, 1, "pixels/view_1", rng_r)
        r_dmap_r = jax.image.resize(dmap_r, (8, 8, 1), method="nearest")

        rgb_l = jp.asarray(rgb[0][..., :3], dtype=jp.float32) / 255.0
        rgb_r = jp.asarray(rgb[1][..., :3], dtype=jp.float32) / 255.0

        info["rng"], rng_noise1, rng_noise2 = jax.random.split(info["rng"], 3)
        rgb_l = distillation.adjust_brightness(
            self.rgb_noise(rng_noise1, rgb_l, info), info["brightness"]
        )
        rgb_r = distillation.adjust_brightness(
            self.rgb_noise(rng_noise2, rgb_r, info), info["brightness"]
        )

        # Required for supervision to stay still.
        socket_pos = data.xpos[self._socket_body]
        dist_from_hidden = jp.linalg.norm(socket_pos[:2] - jp.array([-0.4, 0.33]))
        socket_hidden = jp.where(dist_from_hidden < 3e-2, 1.0, 0.0).reshape(1)

        peg_pos = data.xpos[self._peg_body]
        dist_from_hidden = jp.linalg.norm(peg_pos[:2] - jp.array([0.4, 0.33]))
        peg_hidden = jp.where(dist_from_hidden < 3e-2, 1.0, 0.0).reshape(1)

        obs = {
            "proprio": self._get_proprio(data, info),
            "pixels/view_0": dmap_l,  # view_i for debugging only
            "pixels/view_1": dmap_r,
            "pixels/view_2": rgb_l,
            "pixels/view_3": rgb_r,
            "latent_2": r_dmap_l.ravel(),
            "latent_3": r_dmap_r.ravel(),
            "socket_hidden": socket_hidden,
            "peg_hidden": peg_hidden,
        }
        return obs

    def _get_proprio(self, data: mjx.Data, info: Dict) -> jax.Array:
        """Get the proprio observations for the real sim2real."""
        info["rng"], rng = jax.random.split(info["rng"])
        # qpos_noise = jax.random.uniform(rng, data.qpos.shape) - 0.5
        qpos_noise = jax.random.uniform(
            rng, (16,), minval=0, maxval=self._config.obs_noise.robot_qpos
        )
        qpos_noise = qpos_noise * jp.array(pick_base.QPOS_NOISE_MASK_SINGLE * 2)
        qpos = data.qpos[:16] + qpos_noise
        l_posobs = qpos[self._left_qposadr]
        r_posobs = qpos[self._right_qposadr]

        def dupll(arr):
            # increases size of array by 1 by dupLicating its Last element.
            return jp.concatenate([arr, arr[-1:]])

        assert info["motor_targets"].shape == (14,), print(info["motor_targets"].shape)

        l_velobs = l_posobs - dupll(info["motor_targets"][:7])
        r_velobs = r_posobs - dupll(info["motor_targets"][7:])
        proprio_list = [l_posobs, r_posobs, l_velobs, r_velobs]

        switcher = [info["has_switched"].astype(float).reshape(1)]

        proprio = jp.concat(proprio_list + switcher)
        return proprio

    def add_depth_noise(self, key, img: jp.ndarray):
        """Add realistic depth sensor noise to the depth image."""
        render_width = self._config.vision_config.render_width
        render_height = self._config.vision_config.render_height
        assert img.shape == (render_height, render_width, 1)
        # squeeze
        img = img.squeeze(-1)
        grad_threshold = self._config.obs_noise.grad_threshold
        noise_multiplier = self._config.obs_noise.noise_multiplier

        key_edge_noise, key = jax.random.split(key)
        img = depth_noise.edge_noise(
            key_edge_noise,
            img,
            grad_threshold=grad_threshold,
            noise_multiplier=noise_multiplier,
        )
        key_kinect, key = jax.random.split(key)
        img = depth_noise.kinect_noise(key_kinect, img)
        key_dropout, key = jax.random.split(key)
        img = depth_noise.random_dropout(key_dropout, img)
        key_line, key = jax.random.split(key)
        noise_idx = jax.random.randint(key_line, (), 0, len(self.line_bank))
        img = depth_noise.apply_line_noise(img, self.line_bank[noise_idx])

        # With a low probability, return an all-black image.
        p_blackout = 0.02  # once per 2.5 sec.
        key_blackout, key = jax.random.split(key)
        blackout = jax.random.bernoulli(key_blackout, p=p_blackout)
        img = jp.where(blackout, 0.0, img)

        return img[..., None]

    def process_depth(
        self,
        depth,
        chan: int,
        view_name: str,
        key: Optional[jp.ndarray] = None,
    ):
        """Process depth image with normalization and optional noise."""
        img_size = self._config.vision_config.render_width
        num_cams = len(self._config.vision_config.enabled_cameras)
        assert depth.shape == (num_cams, img_size, img_size, 1)
        depth = depth[chan]
        max_depth = self.max_depth[view_name]
        # max_depth = info['max_depth']
        too_big = jp.where(depth > max_depth, 0, 1)
        depth = depth * too_big
        if self._config.obs_noise.depth and key is not None:
            depth = self.add_depth_noise(key, depth)
        return depth / max_depth  # Normalize

    def rgb_noise(self, key, img, info):
        """Apply domain randomization noise to RGB images."""
        # Assumes images are already normalized.
        pixel_noise = 0.03

        # Add noise to all channels and clip
        key_noise, key = jax.random.split(key)
        noise = jax.random.uniform(key_noise, img.shape, minval=0, maxval=pixel_noise)
        img += noise
        img = jp.clip(img, 0, 1)

        return img

    @property
    def observation_size(self):
        """Return the observation space dimensions for each observation type."""
        # Manually set observation size; default method breaks madrona MJX.
        ret = {
            "has_switched": (1,),
            "proprio": (33,),
            "state": (109,),
            "state_pickup": (106,),
            "peg_hidden": (1,),
            "socket_hidden": (1,),
            "privileged": (110,),
        }
        if self._vision:
            ret.update(
                {
                    "pixels/view_0": (8, 8, 1),
                    "pixels/view_1": (8, 8, 1),
                    "pixels/view_2": (32, 32, 3),
                    "pixels/view_3": (32, 32, 3),
                    "latent_0": (64,),
                    "latent_1": (64,),
                    "latent_2": (64,),
                    "latent_3": (64,),
                }
            )
        else:
            ret["state_with_time"] = (110,)
        return ret


manipulation.register_environment(
    "AlohaPegInsertionVision",
    PegInsertionVision,
    distillation.default_config(),
)
