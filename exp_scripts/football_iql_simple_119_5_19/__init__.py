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


server_name = "whip"

server_list = [(server_name, [0, 1, 2, 3, 4, 5], 1)]

label = "gfootball_simple119_5_19"
config = "qmix"
env_config = "gfootball"

n_repeat = 2

param_dicts = []

shared_params = {
    "t_max": 20000000,
    "epsilon_anneal_time": 2000000,
    "buffer_size": 3000,
    "env_args.episode_limit": 2000,
    "env_args.representation": "ego_simple119",
    "runner": "parallel",
    "batch_size_run": 8,
    "save_model": True,
    "save_model_interval": 250 * 1000,
    "rnn_hidden_dim": 128,
    "local_results_path": "/data/{0}/conlu/results".format(server_name),  # Change server name here
    "env_args.write_full_episode_dumps": True,
    "env_args.write_video": False,
    "env_args.dump_frequency": 100,  # Every # episodes
}

for map_name in ["academy_pass_and_shoot_with_keeper",
                 "academy_run_pass_and_shoot_with_keeper",
                 "academy_3_vs_1_with_keeper"]:
    for rewards in ["scoring,checkpoints", "scoring"]:
        for agent in ['rnn']:
            name = "{0}__{1}__{2}__{3}".format(config, map_name, rewards, agent)
            extend_param_dicts(param_dicts, shared_params,
                               {
                                   "name": name,
                                   "env_args.scenario": map_name,
                                   "env_args.rewards": rewards,
                                   "env_args.logdir": "/data/{0}/conlu/episode_dumps/{1}".format(server_name, name),
                                   # Change server name here
                                   "agent": agent,
                               },
                               repeats=1)
