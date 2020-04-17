import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

if __name__ == '__main__':

    subfolders = [f.path for f in os.scandir('/Users/conglu/Docker Repos/pymarl/results/sacred') if f.is_dir()]

    subfolders.remove('/Users/conglu/Docker Repos/pymarl/results/sacred/_sources')

    label = 'gfootball_iql_full_obs_2'

    label_set = set()

    for metric in ['return_mean', 'score_reward_mean']:
        metric_t = '{0}_T'.format(metric)
        data_dict = defaultdict(list)

        for sf in subfolders:
            with open('{0}/config.json'.format(sf)) as json_file:
                data = json.load(json_file)
                name = data['name']

                label_set.add(data['label'])

                if data['label'] == label:
                    with open('{0}/info.json'.format(sf)) as json_file_2:
                        data = json.load(json_file_2)
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
            plt.savefig("{0}_{1}_{2}.png".format(label, metric, key))

    print(label_set)
