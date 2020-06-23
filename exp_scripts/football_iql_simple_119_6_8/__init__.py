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


server_name = "dgx1"

server_list = [(server_name, [0,1,2,3,4,5,6,7], 2)]

label = "gfootball_simple119_6_8"
config = "qmix"
env_config = "gfootball"

n_repeat = 1

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
    "local_results_path": "/pymarl/results", # Change server name here
    "env_args.write_full_episode_dumps": True,
    "env_args.write_video": True,
    "env_args.dump_frequency": 2000,  # Every # episodes
    "agent": "rnn",
    "target_update_mode": "hard"
}

for map_name in ["academy_3_vs_1_with_keeper"]:
    for rnn_dim in [128, 256]:
        for rewards in ["scoring", "scoring,possession"]:
            for lr in [0.0005, 0.0001]:
                for tar in [100, 200]:
                    name = "{0}__{1}__{2}__{3}__{4}__{5}".format(config, map_name, rewards, rnn_dim, lr, tar)
                    extend_param_dicts(param_dicts, shared_params,
                                       {
                                           "name": name,
                                           "env_args.scenario": map_name,
                                           "env_args.rewards": rewards,
                                           "env_args.logdir": "/pymarl/results/episode_dumps/{0}/{1}".format(label, name),
                                           "rnn_hidden_dim": rnn_dim,
                                           "target_update_interval": tar,
                                           "lr": lr
                                       },
                                       repeats=1)
