from torch import optim
import os


class BaseConfiguration:
    def __init__(self):
        self.patch_name = 'base'
        self.dataset_name = 'debug'
        self.img_dir = os.path.join('..', 'datasets', self.dataset_name)
        self.lab_dir = os.path.join('..', 'datasets', self.dataset_name)
        self.max_labels_per_img = 1
        self.img_size = (112, 112)
        self.patch_size = (256, 256)  # height, width
        self.batch_size = 2
        self.epochs = 10
        self.patch_type = 'l_stripes'
        self.mask_points = [36, 4, 5, 6, 7, 8, 9, 10, 11, 12, 45]
        self.start_learning_rate = 1e-2
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
        self.embedder_name = 'arcface'
        self.embedder_weights_path = os.path.join('..', 'arcface_torch', 'weights', 'arcface_resnet100.pth')
        self.parabola_rate = 0.5
        self.rotation_angle = 0.6
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,  # chin
        # 17, 18, 19, 20, 21,  # left eye-brow
        # 22, 23, 24, 25, 26,  # right eye-brow
        # 27, 28, 29, 30, 31, 32, 33, 34, 35,  # nose
        # 36, 37, 38, 39, 40, 41,  # left eye
        # 42, 43, 44, 45, 46, 47,  # right eye
        # 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,  # outer mouth
        # 61, 62, 63, 64, 65, 66, 67  # inner mouth


class TrainingOnCluster(BaseConfiguration):
    def __init__(self):
        super(TrainingOnCluster, self).__init__()
        self.patch_name = 'cluster'
        self.batch_size = 8


class TrainingOnPrivateComputer(BaseConfiguration):
    def __init__(self):
        super(TrainingOnPrivateComputer, self).__init__()
        self.patch_name = 'private'
        self.batch_size = 1


patch_config_types = {
    "base": BaseConfiguration,
    "cluster": TrainingOnCluster,
    "private": TrainingOnPrivateComputer
}
