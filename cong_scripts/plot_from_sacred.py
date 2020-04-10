import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for metric in ['return_mean', 'score_reward_mean']:
        metric_t = '{0}_T'.format(metric)
        for exp_name in ['iql__academy_pass_and_shoot_with_keeper', 'iql__academy_run_pass_and_shoot_with_keeper', 'iql__academy_counterattack_easy', 'iql__academy_3_vs_1_with_keeper']:
            plt.clf()

            points = []

            for index in range(1, 14):
                with open('/Users/conglu/Docker Repos/pymarl/results/sacred/{0}/config.json'.format(index)) as json_file:
                    data = json.load(json_file)
                    name = data['name']

                with open('/Users/conglu/Docker Repos/pymarl/results/sacred/{0}/info.json'.format(index)) as json_file:
                    data = json.load(json_file)
                    ys = data[metric]
                    xs = data[metric_t]
                    if exp_name in name:
                        points.append((xs, ys))

            for xs, ys in points:
                print(xs)
                plt.plot(xs, ys, label=name)

            plt.xlabel('T env')
            plt.ylabel(metric)
            plt.legend()
            # plt.savefig("{0}_{1}.png".format(metric, exp_name))
