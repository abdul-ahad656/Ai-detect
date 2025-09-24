import json
import os

def load_config(path="config.yaml"):
    import yaml
    with open(path, 'r') as file:
        return yaml.safe_load(file)
