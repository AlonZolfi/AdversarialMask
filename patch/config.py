from torch import optim
import os


class BaseConfiguration:
    def __init__(self):
        self.seed = 42

        # Dataset options
        self.patch_name = 'base'
        self.dataset_name = 'celebA_stripa'
        self.celeb_lab = '2820'  # 2820, 3699, 9040, 9915
        self.is_real_person = False
        self.img_dir = os.path.join('..', 'datasets', self.dataset_name, self.celeb_lab)
        self.train_img_dir = os.path.join('..', 'datasets', self.dataset_name, self.celeb_lab, 'train')
        self.test_img_dir = os.path.join('..', 'datasets', self.dataset_name, self.celeb_lab, 'test')
        self.val_split = 0
        self.test_split = 0.8
        self.shuffle = True
        self.img_size = (112, 112)
        self.batch_size = 2
        self.magnification_ratio = 30

        # Attack options
        self.patch_size = (256, 256)  # height, width
        self.initial_patch = 'l_stripes'  # body, white, random, stripes, l_stripes
        self.epochs = 1
        self.start_learning_rate = 1e-2
        self.es_patience = 5
        self.sc_patience = 1
        self.sc_min_lr = 1e-6
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        patience=self.sc_patience,
                                                                                        min_lr=self.sc_min_lr,
                                                                                        mode='min')

        # Landmark detection options
        self.landmark_detector_type = 'mobilefacenet'  # face_alignment, mobilefacenet
        # Embedder options
        self.embedder_name = 'arcface'  # arcface, vggface2, magface
        self.embedder_weights_path = os.path.join('..', 'face_recognition', 'arcface_torch', 'weights', 'arcface_resnet100.pth')
        self.landmark_folder = os.path.join('../landmark_detection/saved_landmarks',
                                            '_'.join([self.dataset_name, self.embedder_name, str(self.img_size[0])]),
                                            self.celeb_lab)
        self.recreate_landmarks = False
        self.same_person_threshold = 0.4

        # Loss options
        self.dist_loss_type = 'cossim'  # cossim, L2, L1
        self.dist_weight = 0.7
        self.tv_weight = 0.3

        # Test options
        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')

    def set_attribute(self, name, value):
        setattr(self, name, value)


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
