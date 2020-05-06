import json
import numpy as np
from collections import defaultdict
import os

# Parse repeats
# Parse config
# Parse data into config

if __name__ == '__main__':
    directory = '/Users/conglu/Docker Repos/pymarl/docker_text'
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Initially false, flips after each blank line.
            reading_config = False
            config_lines = []
            run_id = -1
            current_info = []
            info_dict = defaultdict(list)
            reading_info = False

            # For pymongo errors
            skip_next_line = False
            full_path = os.path.join(directory, filename)
            print(full_path)
            with open(full_path, "r") as file:
                for line in file:
                    if 'pymongo.errors.WriteError' in line:
                        skip_next_line = True
                    if skip_next_line:
                        skip_next_line = False
                        continue

                    if "{   'action_selector': 'epsilon_greedy'," in line or "'use_tensorboard': False}" in line:
                        # Just a hack
                        if "'use_tensorboard': False}" in line:
                            config_lines.append(line)

                        reading_config = not reading_config
                        if not reading_config:
                            # Finished reading the config portion of the file.
                            json_data = ''.join(config_lines).replace("'", '"').replace('False', 'false').replace('True', 'true').replace('None', 'null')
                            json_parsed = json.loads(json_data)
                            run_id = json_parsed['run_id']

                            print(run_id)

                            if not os.path.isdir('sacred/{}'.format(run_id)):
                                os.mkdir('sacred/{}'.format(run_id))

                            with open('sacred/{}/config.json'.format(run_id), 'w') as outfile:
                                json.dump(json_parsed, outfile, indent=2)
                            config_lines = []
                        if reading_config and not run_id == -1:
                            # Finished reading the info portion of the file.
                            with open('sacred/{}/info.json'.format(run_id), 'w') as outfile:
                                json.dump(info_dict, outfile, indent=2)
                            info_dict = defaultdict(list)

                    if reading_config and not line == '\n':
                        config_lines.append(line)

                    if 'my_main Recent Stats' in line:
                        reading_info = True

                    if ('[' in line or 'Exception in thread' in line) and reading_info and not 'my_main Recent Stats' in line:
                        # End of current config, insert into dict.
                        current_data = ''.join(current_info).replace('\t', ' ').replace('\n', ' ')
                        idx = current_data.find('my_main Recent Stats') + len('my_main Recent Stats | ')
                        current_data = current_data[idx:]
                        current_data = current_data.replace('|', '').replace(':', '').replace('Episode', 'episode')
                        current_data = current_data.split()
                        labels, values = current_data[::2], current_data[1::2]
                        values = [int(v) if '.' not in v else float(v) for v in values]

                        t_env = values[0]

                        for l, v in list(zip(labels, values))[1:]:
                            info_dict[l].append(v)
                            info_dict[l + '_T'].append(t_env)

                        current_info = []
                        reading_info = False

                    if reading_info:
                        current_info.append(line)
                else:
                    # Finished reading the info portion of the file.
                    with open('sacred/{}/info.json'.format(run_id), 'w') as outfile:
                        json.dump(info_dict, outfile, indent=2)
                    info_dict = defaultdict(list)
