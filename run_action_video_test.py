from collections import namedtuple

from absl import flags
from copy import deepcopy
import math
import numpy as np
from operator import attrgetter
import os
import pygame
import sys

import gfootball.env as football_env

if __name__ == '__main__':
    scenario_config = {"11_vs_11_stochastic": {"n_agents": 11},
                       "academy_empty_goal_close": {"n_agents": 1},
                       "academy_empty_goal": {"n_agents": 1},
                       "academy_run_to_score": {"n_agents": 1},
                       "academy_run_to_score_with_keeper": {"n_agents": 1},
                       "academy_pass_and_shoot_with_keeper": {"n_agents": 2},
                       "academy_run_pass_and_shoot_with_keeper": {"n_agents": 2},
                       "academy_3_vs_1_with_keeper": {"n_agents": 3},
                       "academy_corner": {"n_agents": 1},
                       "academy_counterattack_easy": {"n_agents": 4},
                       "academy_single_goal_versus_lazy": {"n_agents": 11}
                       }

    env_name = "academy_3_vs_1_with_keeper"

    env = football_env.create_environment(
        env_name=env_name,
        render=False,
        number_of_left_players_agent_controls=scenario_config[env_name]["n_agents"],
        representation="simple115",
        rewards="scoring",
        write_full_episode_dumps=True,
        write_video=True,
        dump_frequency=1,
        logdir="/pymarl/results/test_videos"
    )

    action_mapping = {
        "idle": 0,
        "left": 1,
        "top_left": 2,
        "top": 3,
        "top_right": 4,
        "right": 5,
        "bottom_right": 6,
        "bottom": 7,
        "bottom_left": 8,
        "long_pass": 9,
        "high_pass": 10,
        "short_pass": 11,
        "shot": 12,
        "sprint": 13,
        "release_direction": 14,
        "release_sprint": 15,
        "sliding": 16,
        "dribble": 17,
        "release_dribble": 18,
    }

    # Alternate method (different actions for each player)
    action_sequence = [['left', 'top', 'bottom']] * 100

    for act_strings in action_sequence:
        actions = [action_mapping[a] for a in act_strings]
        if len(actions) == 1:
            actions = actions[0]
        states, reward, done, info = env.step(actions)

    env.reset()

    # Mirror actions for each player
    action_sequence = (["left"] * 10) + (["top"] * 10) + (["shot"] * 10)
    action_sequence = [[a] * scenario_config[env_name]["n_agents"] for a in action_sequence]
    for act_strings in action_sequence:
        actions = [action_mapping[a] for a in act_strings]
        if len(actions) == 1:
            actions = actions[0]
        states, reward, done, info = env.step(actions)

    env.reset()