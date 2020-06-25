from ..multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert

from absl import flags
from copy import deepcopy
import math
import numpy as np
from operator import attrgetter
import os
import pygame
import sys

import gfootball.env as football_env


class FootballEnv(object):
    def __init__(self, **kwargs):
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)

        # Primary config
        self.scenario = getattr(args, "scenario", "11_vs_11_stochastic")
        self.game_visibility = getattr(args, "game_visibility", "full")
        self.n_actions = 19  # hard-coded
        self.representation = getattr(args, "representation", "simple115")
        self.full_obs_flag = getattr(args, "full_obs", False)
        self.view_angle = getattr(args, "view_angle", 160)
        self.rewards = getattr(args, "rewards", "scoring")

        # Video dumping config
        self.write_full_episode_dumps = getattr(args, "write_full_episode_dumps", False)
        self.write_video = getattr(args, "write_video", False)
        self.dump_frequency = getattr(args, "dump_frequency", 1)
        self.logdir = getattr(args, "logdir", "episode_dumps")

        # Environment modifiers
        self.move_goalkeeper = getattr(args, "move_goalkeeper", False)
        self.difficulty_override = getattr(args, "env_difficulty", -1)

        # Secondary config
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
        if getattr(args, "n_agents", -1) == -1:
            self.n_agents = scenario_config[self.scenario]["n_agents"]
        else:
            assert args.n_agents <= scenario_config[self.scenario]["n_agents"], \
                "Scenario only supports up to {} agents - you supplied {}!".format(
                    scenario_config[self.scenario]["n_agents"], args.n_agents)
            self.n_agents = args.n_agents

        self.episode_limit = args.episode_limit if getattr(args, "episode_limit",
                                                           -1) != -1 else 1000  # TODO: Look up correct episode length!
        self.observation_reference_frame = getattr(args, "observation_reference_frame", "fixed")

        self.env = football_env.create_environment(
            env_name=self.scenario,
            render=False,
            number_of_left_players_agent_controls=self.n_agents,
            representation=self.representation,
            rewards=self.rewards,
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_video=self.write_video,
            dump_frequency=self.dump_frequency,
            logdir=self.logdir,
            # po_view_cone_xy_opening=self.view_angle,
            # full_obs_flag=self.full_obs_flag,
        )

        self.reset()

        self.obs_size = self.observations[0].shape

        self.state = self.env.get_global_state()
        self.state_size = self.state.shape

    def _make_ma_obs(self, obs):
        """
        Convert observation from gfootball_env into a multi-agent observation.
        :param obs:
        :param env:
        :return:
        """
        if self.representation in ["pixels", "pixels_gray", "extracted"]:
            obs = np.moveaxis(obs, -1, 0)
        if self.game_visibility == "full" and self.observation_reference_frame == "fixed":
            observations = [deepcopy(obs) for _ in range(self.n_agents)]
        else:
            raise NotImplementedError
        return observations

    def _make_state(self, state):
        """
        Convert observation from gfootball_env into a multi-agent observation.
        :param obs:
        :param env:
        :return:
        """
        if self.representation in ["pixels", "pixels_gray", "extracted"]:
            state = np.moveaxis(state, -1, 0)
        return state

    def step(self, actions):
        """ Returns reward, terminated, info """
        if not self.done:
            # Convert pytorch tensor to list (expand if single action)
            actions = actions.tolist()
            # actions = (actions.data).cpu().numpy()
            if len(actions) == 1:
                actions = actions[0]

            states, reward, done, info = self.env.step(actions)
            self.done = done
            self.state = info['global_state']

            if len(states.shape) == 1:
                # Single observation, this just duplicates states
                self.observations = self._make_ma_obs(states)
            else:
                # Many observations
                # TODO: do the moveaxis conversion on this later (states is an ndarray)
                self.observations = states

            self.steps += 1
            if self.episode_limit != -1 and self.steps == self.episode_limit:
                self.done = True

            return reward if isinstance(reward, np.float32) else reward[0], done, info
        else:
            return 0, True, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.observations

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.observations[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_size

    def get_state(self):
        # Global state passed through info
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state_size

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        # Mask for the actions
        return np.ones((self.n_actions,))

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        stats = {}
        return stats

    # TODO: Temp hack
    def get_agg_stats(self, stats):
        return {}

    def reset(self, test_mode=False, t_env=-1):
        states = self.env.reset(move_goalkeeper=self.move_goalkeeper, difficulty_override=self.difficulty_override,
                                test_mode=test_mode, t_env=t_env)
        if len(states.shape) == 1:
            # Single observation
            # This just duplicates states
            self.observations = self._make_ma_obs(states)
        else:
            # Many observations
            # TODO: do the moveaxis conversion on this later
            self.observations = states

        self.done = False
        self.steps = 1
        return

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
