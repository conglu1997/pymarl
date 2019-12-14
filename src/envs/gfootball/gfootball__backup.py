from ..multiagentenv import MultiAgentEnv
from collections import OrderedDict
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

        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)

        # Primary config
        self.scenario = getattr(args, "scenario", "11_vs_11_stochastic")
        self.game_visibility = getattr(args, "game_visibility", "full")
        self.n_actions = 20  # hard-coded

        # Secondary config
        scenario_config = {"11_vs_11_stochastic": {"n_agents":11},
                           "academy_empty_goal_close": {"n_agents":1},
                           "academy_empty_goal": {"n_agents":1},
                           "academy_run_to_score": {"n_agents": 1},
                           "academy_run_to_score_with_keeper": {"n_agents": 1},
                           "academy_pass_and_shoot_with_keeper": {"n_agents": 2},
                           "academy_run_pass_and_shoot_with_keeper": {"n_agents": 2},
                           "academy_3_vs_1_with_keeper": {"n_agents":3},
                           "academy_corner": {"n_agents": 1},
                           "academy_counterattack_easy": {"n_agents": 4},
                           "academy_single_goal_versus_lazy": {"n_agents": 11}
                           }
        if getattr(args, "n_agents", -1) == -1:
            self.n_agents = scenario_config[self.scenario]["n_agents"]
        else:
            assert args.n_agents <= scenario_config[self.scenario]["n_agents"], \
                "Scenario only supports up to {} agents - you supplied {}!".format(scenario_config[self.scenario]["n_agents"], args.n_agents)
            self.n_agents = args.n_agents

        self.episode_limit = args.episode_limit if getattr(args, "episode_limit", -1) != -1 else 25  # TODO: Look up correct episode length!
        self.observation_reference_frame = getattr(args, "observation_reference_frame", "fixed")

        self.reset()
        pass

    def _make_ma_obs(self, obs, env):
        """
        Convert observation from gfootball_env into a multi-agent observation.
        :param obs:
        :param env:
        :return:
        """
        if self.game_visibility == "full" and self.observation_reference_frame == "fixed":
            observations = [deepcopy(obs) for _ in range(self.n_agents)]
        else:
            raise NotImplementedError
        return observations

    def _make_state(self, obs, env):
        """
        Convert observation from gfootball_env into a multi-agent observation.
        :param obs:
        :param env:
        :return:
        """
        return obs

    def step(self, actions):
        """ Returns reward, terminated, info """
        if not self.done:
            action = self.env.action_space.sample()
            states, reward, done, info = self.env.step(action)
            state = states if len(states.shape) == 1 else states[0]
            self.done = done
            self.observations = self._make_ma_obs(state, self.env)
            self.state = self._make_state(state, self.env)
            self.steps +=1
            if self.episode_limit != -1 and self.steps == self.episode_limit:
                self.done = True
            return reward if isinstance(reward, np.float32) else reward[0], done, info
        else:
            return 0, True, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return OrderedDict([("1d", self.observations)])

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.observations[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_size

    def get_state(self):
        return OrderedDict([("1d", self.state)])  # DEBUG

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state_size

    def get_avail_actions(self):
        return [np.ones((self.n_actions,)) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
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

    def reset(self):
        self.env = football_env.create_environment(
            env_name = self.scenario,
            render = False,
            number_of_left_players_agent_controls = self.n_agents,
            representation="simple115")
        states = self.env.reset()
        state = states if len(states.shape) == 1 else states[0]
        self.observations = self._make_ma_obs(state, self.env)
        self.obs_size = self.observations[0].shape
        self.state = self._make_state(state, self.env)
        self.state_size = self.state.shape
        self.done = False
        self.steps = 1
        return

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": OrderedDict([("1d", self.get_state_size())]),
                    "obs_shape":  OrderedDict([("1d", self.get_obs_size())]),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
