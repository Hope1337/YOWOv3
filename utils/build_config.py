import yaml

def build_config(config_file='config/ava_config.yaml'):
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    if config['active_checker']:
        pass
    
    return config