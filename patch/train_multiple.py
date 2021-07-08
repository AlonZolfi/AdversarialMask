from config import patch_config_types
from train import AdversarialMask
from test import Evaluator


def train_multiple_seeds():
    mode = 'private'
    config = patch_config_types[mode]()
    for test_split in [0.99, 0.95, 0.92, 0.9, 0.87, 0.85, 0.8, 0.72, 0.58]:
        config.set_attribute('test_split', test_split)
        for seed in [42, 43, 44, 45, 46]:
            config.set_attribute('seed', seed)
            adv_mask = AdversarialMask(config)
            adv_mask.train()
            evaluator = Evaluator(adv_mask)
            evaluator.test()


train_multiple_seeds()
