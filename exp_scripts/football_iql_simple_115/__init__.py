import copy


def extend_param_dicts(param_dicts, shared_params, exp_params, repeats=1):
    # copy
    result = copy.deepcopy(shared_params)
    # add experiment parameters to shared params
    result.update(exp_params)
    for i in range(repeats):
        # result.update({"par_id": [i]})
        param_dicts.append(copy.deepcopy(result))
    return param_dicts


server_list = [("gimli", [1, 2, 3, 4], 4)]

label = "gfootball_iql_simple115_22_4"
config = "iql"
env_config = "gfootball"

n_repeat = 2

param_dicts = []

shared_params = {
    "t_max": 20000000,
    "epsilon_anneal_time": 250000,  # Should I change this?
    "buffer_size": 5000,
    "env_args.episode_limit": 1000,
    "env_args.representation": "simple115",
    "agent": "ff",
    "obs_last_action": False,
    "runner": "parallel",
    "batch_size_run": 8,
    "save_model": True,
    "save_model_interval": 250 * 1000,
    "rnn_hidden_dim": 128,
    "local_results_path": "/data/gimli/conlu/results",  # Change server name here
    "write_full_episode_dumps": False,
    "write_video": False,
    "dump_frequency": 1,
    "logdir": "/data/gimli/conlu/episode_dumps",  # Change server name here
}

for map_name in ["academy_empty_goal",
                 "academy_empty_goal_close",
                 "academy_run_to_score",
                 "academy_run_to_score_with_keeper",
                 "academy_pass_and_shoot_with_keeper",
                 "academy_run_pass_and_shoot_with_keeper",
                 "academy_3_vs_1_with_keeper",
                 "academy_counterattack_easy"]:
    for rewards in ["scoring,checkpoints", "scoring"]:
        for agent in ['ff', 'rnn']:
            extend_param_dicts(param_dicts, shared_params,
                               {
                                   "name": "iql__{0}__{1}__{2}".format(map_name, rewards, agent),
                                   "env_args.scenario": map_name,
                                   "env_args.rewards": rewards,
                                   "agent": agent,
                               },
                               repeats=1)
