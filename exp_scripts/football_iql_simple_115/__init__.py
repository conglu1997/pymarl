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

label = "gfootball_iql_simple115_20_4"
config = "iql"
env_config = "gfootball"

n_repeat = 1

param_dicts = []

shared_params = {
    "t_max": 2000000,
    "epsilon_anneal_time": 250000,
    "buffer_size": 5000,
    "env_args.episode_limit": 1000,
    "env_args.representation": "simple115",
    # "env_args.full_obs": True,
    "env_args.rewards": "scoring,checkpoints",
    "agent": "ff",
    "obs_last_action": False
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
        extend_param_dicts(param_dicts, shared_params,
                           {
                               "name": "iql__{0}__{1}".format(map_name, rewards),
                               "env_args.scenario": map_name,
                               "env_args.rewards": rewards,
                           },
                           repeats=1)
