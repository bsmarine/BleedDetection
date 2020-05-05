import sys
import subprocess
import os
import pickle
import json
import importlib.util

def write_config_params(config_path):

    cf_file = import_module('cf_file',config_path)

    cf = cf_file.configs(None)

    print ("Root Dir: "+cf.root_dir)

def import_module(name, path):
    """
    correct way of importing a module dynamically in python 3.
    :param name: name given to module instance.
    :param path: path to module.
    :return: module: returned module instance.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":

  config_path = '/home/marinb02/mdtk/medicaldetectiontoolkit/experiments/toy_exp/configs.py'

  write_config_params(config_path)
