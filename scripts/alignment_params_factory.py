import json
from itertools import product

with open("params_config.json", "r") as f:
    config = json.load(f)

defaults = config["defaults"]
bucketing_section = config["bucketing"]
enabled_bucketings = config.get("enabled_bucketings", list(bucketing_section.keys()))

def build_param_grid(bucketing_value):
    grid = defaults.copy()
    overrides = bucketing_section.get(bucketing_value, {})
    grid.update(overrides)
    grid["bucketing"] = [bucketing_value]
    return grid

def generate_valid_combinations():
    for bucketing_value in enabled_bucketings:
        grid = build_param_grid(bucketing_value)
        keys = list(grid.keys())
        values = list(grid.values())
        for combination in product(*values):
            params = dict(zip(keys, combination))
            yield params

#for i, params in enumerate(generate_valid_combinations(), 1):
#    print(f"{i}: {params}")
