from config import patch_config_types
from train import AdversarialMask
from test import Evaluator


def train_multiple_seeds():
    mode = 'private'
    config = patch_config_types[mode]()
    output_folders = []
    for num_of_train_images in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]:
        config.set_attribute('num_of_train_images', num_of_train_images)

        for seed in [42, 43, 44]:
            config.set_attribute('seed', seed)

            config.update_current_dir()
            adv_mask = AdversarialMask(config)
            adv_mask.train()
            evaluator = Evaluator(adv_mask)
            evaluator.test()
            output_folders.append(config.current_dir)

    print(output_folders)


train_multiple_seeds()
