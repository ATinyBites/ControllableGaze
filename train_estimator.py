import torch
from gaze_estimation.trainer import Trainer
from gaze_estimation.config import get_config
from gaze_estimation.data_loader import get_train_loader, get_test_loader
import random
import numpy as np


def run(config):
    kwargs = {}
    if config.use_gpu:
        kwargs = {'num_workers': config.num_workers}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_loader(
            config.data_dir, config.batch_size, is_shuffle=True,
            **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.batch_size, is_shuffle=False,
            **kwargs
        )
    # instantiate trainer
    trainer = Trainer(config, data_loader) 

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

if __name__ == '__main__':
    # ensure reproducibility
    seed = 4
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    config, unparsed = get_config()
    run(config)
 
