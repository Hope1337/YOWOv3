from datasets.ucf.load_data import build_ucf_dataset


def build_dataset(config, phase):
    dataset = config['dataset']

    if dataset == 'ucf':
        return build_ucf_dataset(config, phase)