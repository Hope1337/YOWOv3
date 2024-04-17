from utils.build_config import build_config


config = build_config()
print(config['idx2name'])

idx2name = config['idx2name']
print(idx2name[1])