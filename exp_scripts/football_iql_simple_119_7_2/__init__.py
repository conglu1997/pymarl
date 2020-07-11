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


# server_name = "draco"
# server_list = [(server_name, [4, 5, 6], 2)]

server_name = "orion"
server_list = [(server_name, [4, 5, 6, 7], 2)]

# server_name = "whip"
# server_list = [(server_name, [0,1, 2, 3, 4, 5, 6, 7], 1)]

label = "gfootball_simple119_6_28"
config = "qmix"
# config = "iql"
env_config = "gfootball"

n_repeat = 2

param_dicts = []

# Should use this for dgx1: -v $(readlink -f /raid/data/):/data/


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
    "local_results_path": "/pymarl/results" if server_name == 'dgx1' else "/data/{0}/conlu/results".format(server_name),
    "env_args.write_full_episode_dumps": True,
    "env_args.write_video": True,
    "env_args.dump_frequency": 200,  # Every # episodes
    "agent": "rnn",
    "env_args.move_goalkeeper": False,
    "env_args.env_difficulty": 0.6,
    # 2.5x normal rate
    "log_interval": 5000,
    "runner_log_interval": 5000,
    "learner_log_interval": 5000,
}

# Changing difficulty (8 exps) - ON ORION
for map_name in ["academy_3_vs_1_with_keeper"]:
    for rnn_dim in [128]:
        for rewards in ["scoring", "scoring,possession"]:
            for lr in [0.0005, 0.0001]:
                for tar in [100, 200]:
                    for env_difficulty in [0.0]:
                        name = "{0}__{1}__{2}__{3}__{4}__{5}__diff{6}".format(config, map_name, rewards, rnn_dim, lr, tar, env_difficulty)
                        extend_param_dicts(param_dicts, shared_params,
                                           {
                                               "name": name,
                                               "env_args.scenario": map_name,
                                               "env_args.rewards": rewards,
                                               "env_args.logdir": "/pymarl/results/episode_dumps/{0}/{1}".format(label,
                                                                                                                 name) if server_name == 'dgx1' else "/data/{0}/conlu/episode_dumps/{1}/{2}".format(
                                                   server_name, label, name),
                                               "rnn_hidden_dim": rnn_dim,
                                               "target_update_mode": "hard",
                                               "target_update_interval": tar,
                                               "lr": lr,
                                               "env_args.move_goalkeeper": False,
                                               "env_args.env_difficulty": env_difficulty,
                                           },
                                           repeats=1)
#
# # New 3vs1 positions (6 exps)
# for map_name in ["academy_3_vs_1_with_keeper"]:
#     for rnn_dim in [128]:
#         for rewards in ["scoring", "scoring,possession"]:
#             for lr in [0.0005, 0.0003, 0.0001]:
#                 for tar in [100]:
#                     name = "{0}__{1}__{2}__{3}__{4}__{5}__newpos".format(config, map_name, rewards, rnn_dim, lr, tar)
#                     extend_param_dicts(param_dicts, shared_params,
#                                        {
#                                            "name": name,
#                                            "env_args.scenario": map_name,
#                                            "env_args.rewards": rewards,
#                                            "env_args.logdir": "/pymarl/results/episode_dumps/{0}/{1}".format(label,
#                                                                                                              name) if server_name == 'dgx1' else "/data/{0}/conlu/episode_dumps/{1}/{2}".format(
#                                                server_name, label, name),
#                                            "rnn_hidden_dim": rnn_dim,
#                                            "target_update_mode": "hard",
#                                            "target_update_interval": tar,
#                                            "lr": lr,
#                                            "env_args.move_goalkeeper": True,
#                                            "env_args.env_difficulty": 0.6,
#                                        },
#                                        repeats=1)


# # Empty goal, trying to get 1 reward (do this IQL instead) (8 experiments) - ON WHIP
# for map_name in ["academy_empty_goal_close"]:
#     for rnn_dim in [128, 256]:
#         for rewards in ["scoring"]:
#             for lr in [0.0005, 0.0001]:
#                 for tar in [100, 200]:
#                     name = "{0}__{1}__{2}__{3}__{4}__{5}__eg".format(config, map_name, rewards, rnn_dim, lr, tar)
#                     extend_param_dicts(param_dicts, shared_params,
#                                        {
#                                            "name": name,
#                                            "env_args.scenario": map_name,
#                                            "env_args.rewards": rewards,
#                                            "env_args.logdir": "/pymarl/results/episode_dumps/{0}/{1}".format(label,
#                                                                                                              name) if server_name == 'dgx1' else "/data/{0}/conlu/episode_dumps/{1}/{2}".format(
#                                                server_name, label, name),
#                                            "rnn_hidden_dim": rnn_dim,
#                                            "target_update_interval": tar,
#                                            "lr": lr,
#                                            "env_args.move_goalkeeper": False,
#                                            "env_args.env_difficulty": 0.6,
#                                        },
#                                        repeats=1)
