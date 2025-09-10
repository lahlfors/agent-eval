"""Initializes and configures the web shopping simulation environment.

This module is responsible for setting up the `WebAgentTextEnv`, a custom Gym
environment that simulates a text-based web shopping experience. It registers
the environment with Gym, provides a function to initialize it, and creates a
global instance of the environment (`webshop_env`) that can be imported and
used by the agent's tools.
"""

# Copyright 2025 Google LLC
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

import gym

# Register the custom web environment with Gym if it's not already registered.
gym.envs.registration.register(
    id="WebAgentTextEnv-v0",
    entry_point=(
        "personalized_shopping.shared_libraries.web_agent_site.envs.web_agent_text_env:WebAgentTextEnv"
    ),
)

def init_env(num_products: int) -> gym.Env:
    """Initializes the web shopping Gym environment.

    Args:
        num_products: The number of products to load into the environment's
                      database.

    Returns:
        An instance of the WebAgentTextEnv Gym environment.
    """
    env = gym.make(
        "WebAgentTextEnv-v0",
        observation_mode="text",
        num_products=num_products,
    )
    return env

