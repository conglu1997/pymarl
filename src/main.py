import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml
import pymongo

from run import run

SETTINGS['CAPTURE_MODE'] = "fd"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

# results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# Append results to data instead
results_path = os.path.join("/data", str(os.environ.get('STORAGE_HOSTNAME')), "conlu", "results")

mongo_client = None


def setup_mongodb():
    # The central mongodb for our deepmarl experiments
    # You need to set up local port forwarding to ensure this local port maps to the server
    # if conf_str == "":
    # db_host = "localhost"
    # db_port = 27027 # Use a different port from the default mongodb port to avoid a potential clash

    db_url = r'mongodb://pymarlOwner:EMC7Jp98c8rE7FxxN7g82DT5spGsVr9A@gandalf.cs.ox.ac.uk:27017/pymarl'
    db_name = r'pymarl'

    client = None

    # First try to connect to the central server. If that doesn't work then just save locally
    maxSevSelDelay = 10000  # Assume 1ms maximum server selection delay
    try:
        # Check whether server is accessible
        logger.info("Trying to connect to mongoDB '{}'".format(db_url))
        client = pymongo.MongoClient(db_url, ssl=True, serverSelectionTimeoutMS=maxSevSelDelay)
        client.server_info()
        # If this hasn't raised an exception, we can add the observer
        ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name, ssl=True))  # db_name=db_name,
        logger.info("Added MongoDB observer on {}.".format(db_url))
    except pymongo.errors.ServerSelectionTimeoutError:
        logger.warning("Couldn't connect to MongoDB.")
        logger.info("Fallback to FileStorageObserver in results/sacred.")

    return client


@ex.main
def my_main(_run, _config, _log):
    global mongo_client

    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(_config["seed"])
    th.manual_seed(_config["seed"])
    config['env_args']['seed'] = config["seed"]

    # Setting run_id in config
    _config["run_id"] = _run._id

    # run the framework
    run(_run, _config, _log, mongo_client)

    # force exit
    os._exit(0)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # now add all the config to sacred
    ex.add_config(config_dict)

    mongo_client = setup_mongodb()

    # Save to disk by default for sacred, even if we are using the mongodb
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)
