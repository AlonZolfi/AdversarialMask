from torch import optim
import os
import datetime
import time


class BaseConfiguration:
    def __init__(self):
        self.seed = 42

        # Dataset options
        self.patch_name = 'base'
        self.dataset_name = 'CelebA_aligned'  # CASIA-WebFace_aligned_100, CASIA-WebFace_aligned_1000, CelebA_aligned
        self.is_real_person = False
        self.img_dir = os.path.join('..', 'datasets', self.dataset_name)
        self.number_of_people = 100
        self.celeb_lab = os.listdir(self.img_dir)[:self.number_of_people]  # 2820, 3699, 9040, 9915, os.listdir(self.img_dir)
        self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}
        self.train_img_dir = os.path.join('..', 'datasets', self.dataset_name, 'train')
        self.test_img_dir = os.path.join('..', 'datasets', self.dataset_name, 'test')
        self.num_of_train_images = 5
        self.val_split = 0
        self.test_split = 0.8
        self.shuffle = True
        self.img_size = (112, 112)
        self.train_batch_size = 2
        self.test_batch_size = 8
        self.magnification_ratio = 30

        # Attack options
        self.mask_aug = True
        self.patch_size = (256, 256)  # height, width
        self.initial_patch = 'white'  # body, white, random, stripes, l_stripes
        self.epochs = 0
        self.start_learning_rate = 1e-2
        self.es_patience = 7
        self.sc_patience = 2
        self.sc_min_lr = 1e-6
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        patience=self.sc_patience,
                                                                                        min_lr=self.sc_min_lr,
                                                                                        mode='min')

        # Landmark detection options
        self.landmark_detector_type = 'mobilefacenet'  # face_alignment, mobilefacenet
        # Embedder options

        self.embedder_name = ['arcface', 'magface']  # arcface, vggface2, magface, shpereface
        self.embedder_weights_path = [os.path.join('..', 'face_recognition', 'arcface_torch', 'weights', 'arcface_resnet100.pth'),
                                      os.path.join('..', 'face_recognition', 'magface_torch', 'weights', 'magface_resnet100.pth')]
        # self.landmark_folder = os.path.join('../landmark_detection/saved_landmarks',
        #                                     '_'.join([self.dataset_name, self.embedder_name, str(self.img_size[0])]),
        #                                     self.celeb_lab)
        self.recreate_landmarks = False
        self.same_person_threshold = 0.55

        # Loss options
        self.dist_loss_type = 'cossim'  # cossim, L2, L1
        self.dist_weight = 0.99
        self.tv_weight = 0.01

        # Test options
        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')

        self.update_current_dir()

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = "experiments/" + month_name + '/' + time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S")
        if 'SLURM_JOBID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOBID']


class TrainingOnCluster(BaseConfiguration):
    def __init__(self):
        super(TrainingOnCluster, self).__init__()
        self.patch_name = 'cluster'


class TrainingOnPrivateComputer(BaseConfiguration):
    def __init__(self):
        super(TrainingOnPrivateComputer, self).__init__()
        self.patch_name = 'private'


patch_config_types = {
    "base": BaseConfiguration,
    "cluster": TrainingOnCluster,
    "private": TrainingOnPrivateComputer
}
