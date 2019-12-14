from functools import partial

from .multiagentenv import MultiAgentEnv
from .gfootball import FootballEnv

def env_fn(env, **kwargs) -> MultiAgentEnv: 
    return env(**kwargs)


REGISTRY = {}
REGISTRY["gfootball"] = partial(env_fn, env=FootballEnv)

