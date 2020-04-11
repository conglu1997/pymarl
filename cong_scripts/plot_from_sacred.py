import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == '__main__':
    for metric in ['return_mean', 'score_reward_mean']:
        metric_t = '{0}_T'.format(metric)
        data_dict = defaultdict(list)

        for index in range(88821, 88844):
            with open('/Users/conglu/Docker Repos/pymarl/results/sacred/{0}/config.json'.format(index)) as json_file:
                data = json.load(json_file)
                name = data['name']
                print(name)

            with open('/Users/conglu/Docker Repos/pymarl/results/sacred/{0}/info.json'.format(index)) as json_file:
                data = json.load(json_file)
                ys = data[metric]
                xs = data[metric_t]
                data_dict[name].append((xs, ys))

        for key, points in data_dict.items():
            plt.clf()
            for xs, ys in points:
                plt.plot(xs, ys, label=name)
            plt.xlabel('T env')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig("{0}_{1}.png".format(metric, key))
