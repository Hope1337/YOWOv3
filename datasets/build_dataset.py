from datasets.ucf.load_data import build_ucf_dataset
from datasets.ava.load_data import build_ava_dataset
from datasets.jhmdb.load_data import build_jhmdb_dataset

def build_dataset(config, phase):
    dataset = config['dataset']

    if dataset == 'ucf':
        return build_ucf_dataset(config, phase)
    elif dataset == 'ava':
        return build_ava_dataset(config, phase)
    elif dataset == 'jhmdb':
        return build_jhmdb_dataset(config, phase)