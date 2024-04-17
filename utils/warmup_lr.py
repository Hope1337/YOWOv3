class LinearWarmup():
    def __init__(self, config):
        self.max_step = config['max_step_warmup']
        self.lr_init  = config['lr']
    
    def __call__(self, optimizer, step):
        
        if (step > self.max_step):
            return
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr_init * step / self.max_step
