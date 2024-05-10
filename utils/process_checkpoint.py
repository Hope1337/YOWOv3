import torch

def load_checkpoint(path):
    checkpoint_dict = torch.load(path)
    config          = checkpoint_dict['config']
    state_dict      = checkpoint_dict['state_dict']
    ema_state_dict  = checkpoint_dict['ema_state_dict']
    meta_data       = checkpoint_dict['meta_data']

    return config, state_dict, ema_state_dict, meta_data


def save_checkpoint(config, model, ema_model, cur_epoch, save_path):
    state_dict      = model.state_dict
    ema_state_dict  = ema_model.ema.state_dict
    meta_data       = {'ema_updates' : ema_model.updates,
                       'cur_epoch'   : cur_epoch}
    
    checkpoint_dict = {'config'         : config,
                       'state_dict'     : state_dict,
                       'ema_state_dict' : ema_state_dict,
                       'meta_data'      : meta_data}
    
    torch.save(checkpoint_dict, save_path)