from face_alignment import FaceAlignment, LandmarksType
import torch
from utils import SplitDataset, CustomDataset, CustomDataset1, load_embedder, EarlyStopping
import os

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision import transforms


def save_to_file():
    dataset_name = 'celebA/2820'
    img_dir = os.path.join('..', 'datasets', dataset_name)
    img_size = (112, 112)
    face_align = FaceAlignment(LandmarksType._2D, device=str(device))
    custom_dataset = CustomDataset1(img_dir=img_dir,
                                    img_size=img_size,
                                    transform=transforms.Compose(
                                        [transforms.Resize(img_size), transforms.ToTensor()]))

    train_loader, val_loader, test_loader = SplitDataset(custom_dataset)(
        val_split=0.45,
        test_split=0.05,
        shuffle=True,
        batch_size=1)

    folder = 'landmarks/celebA/2820'

    for loader in [train_loader, val_loader, test_loader]:
        for image, img_name in loader:
            points = face_align.get_landmarks_from_batch(image * 255)
            single_face_points = [landmarks[:68] for landmarks in points]
            preds = torch.tensor(single_face_points, device=device)
            torch.save(preds, os.path.join(folder, img_name[0].replace('jpg', 'pt')))


save_to_file()