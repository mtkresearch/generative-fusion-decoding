import yaml
import argparse
from collections import namedtuple

def process_config(file_path, args=argparse.Namespace()):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    for arg in vars(args):
        if getattr(args, arg) is not None and arg in config_dict.keys():
            config_dict[arg] = getattr(args, arg)
    Config = namedtuple('Config', config_dict.keys())
    return  Config(**config_dict)

def combine_config(config1, config2):
    CombinedConfig = namedtuple('Config', config1._fields + config2._fields)
    return CombinedConfig(*(config1+config2))