from config import patch_config_types
from train import AdversarialMask
from test import Evaluator
import os
from shutil import move
import pickle
from pathlib import Path
import datetime
import time
import torch


def train_multiple_persons():
    mode = 'targeted'
    config = patch_config_types[mode]()
    output_folders = []
    for i, lab in enumerate(sorted(os.listdir(config.train_img_dir)[:100])):
        config.update_current_dir()
        config.set_attribute('celeb_lab', [lab])
        config.set_attribute('celeb_lab_mapper', {0: lab})
        config.update_test_celeb_lab()
        adv_mask = AdversarialMask(config)
        print(f'Starting train person {i+1}...', flush=True)
        adv_mask.train()
        print('Finished train...', flush=True)
        evaluator = Evaluator(adv_mask)
        print(f'Starting test person {i+1}...', flush=True)
        evaluator.test()
        print('Finished test...', flush=True)
        output_folders.append(config.current_dir)
        del adv_mask, evaluator
        torch.cuda.empty_cache()


if __name__ == '__main__':
    train_multiple_persons()
