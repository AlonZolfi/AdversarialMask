import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time
import os
import shutil
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object


save_path = 'classification_model.pth'

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # binary classification (num_of_class == 2)
model.load_state_dict(torch.load(save_path))
model.to(device)

model.eval()
start_time = time.time()
# count_men = 0
# count_women = 0
with torch.no_grad():
    for folder_name in tqdm(os.listdir('../datasets/CASIA-WebFace_aligned')[:1000]):
        image_path = os.listdir('../datasets/CASIA-WebFace_aligned/' + folder_name)[0]
        img = Image.open('../datasets/CASIA-WebFace_aligned/' + folder_name + '/' + image_path)
        img_t = transforms.ToTensor()(img).unsqueeze(0)
        img_t = img_t.to(device)

        outputs = model(img_t)
        _, preds = torch.max(outputs, 1)
        if preds.item() == 0:
            # count_women += 1
            shutil.copytree('../datasets/CASIA-WebFace_aligned/' + folder_name, '../datasets/CASIA-Women/' + folder_name)
        else:
            # count_men +=1
            shutil.copytree('../datasets/CASIA-WebFace_aligned/' + folder_name, '../datasets/CASIA-Men/' + folder_name)


# print(count_men)
# print(count_women)
