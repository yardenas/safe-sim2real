# Copyright 2024 The Brax Authors.
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

"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""

from typing import TypeAlias

from brax.training import types

from ss2r.algorithms.sac.losses import make_losses as sac_make_losses
from ss2r.algorithms.sac.networks import SafeSACNetworks

Transition: TypeAlias = types.Transition


def make_losses(
    sac_network: SafeSACNetworks,
    reward_scaling: float,
    cost_scaling: float,
    discounting: float,
    safety_discounting: float,
    action_size: int,
    use_bro: bool,
    init_alpha: float | None,
):
    alpha_loss, critic_loss, actor_loss = sac_make_losses(
        sac_network,
        reward_scaling,
        cost_scaling,
        discounting,
        safety_discounting,
        action_size,
        use_bro,
        init_alpha,
    )
    return alpha_loss, critic_loss, actor_loss
