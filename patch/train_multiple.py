from config import patch_config_types
from train import AdversarialMask
from test import Evaluator
import os
import pickle


def train_multiple():
    mode = 'private'
    config = patch_config_types[mode]()
    output_folders = []
    for lab in os.listdir(config.img_dir):
        config.update_current_dir()
        config.set_attribute('celeb_lab', [lab])
        config.set_attribute('celeb_lab_mapper', {0: lab})
        adv_mask = AdversarialMask(config)
        adv_mask.train()
        evaluator = Evaluator(adv_mask)
        evaluator.test()
        output_folders.append(config.current_dir)
    job_id = ''
    if 'SLURM_JOBID' in os.environ.keys():
        job_id = os.environ['SLURM_JOBID']
    with open(job_id + 'output_folders.pickle', 'wb') as f:
        pickle.dump(output_folders, f)
    print(output_folders)


train_multiple()
