from torch import optim
import os


class BaseConfiguration:
    def __init__(self):
        # Dataset options
        self.patch_name = 'base'
        self.dataset_name = 'celebA_stripa'
        self.celeb_lab = '2820'
        self.img_dir = os.path.join('..', 'datasets', self.dataset_name, self.celeb_lab)
        self.val_split = 0.2
        self.test_split = 0.6
        self.shuffle = True
        self.img_size = (112, 112)
        self.batch_size = 2

        # Attack options
        self.patch_size = (256, 256)  # height, width
        self.initial_patch = 'white'  # body, white, random, stripes, l_stripes
        self.epochs = 100
        self.start_learning_rate = 5e-3
        self.es_patience = 5
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        factor=0.7,
                                                                                        patience=0,
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
        self.dist_weight = 1
        self.tv_weight = 0

        # Test options
        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')


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
