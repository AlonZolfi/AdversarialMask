import math
from PIL import Image, ImageDraw

x, y = 256, 256
eX, eY = 350, 100
bbox = (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2)

# creating new Image object
img = Image.new("RGB", (x, y))

# create ellipse image
img1 = ImageDraw.Draw(img)
img1.ellipse(bbox, fill="white")

import torchvision
img_t = torchvision.transforms.ToTensor()(img)
img_t[:,:, :40] = 0
img_t[:,:, -40:] = 0
torchvision.transforms.ToPILImage()(img_t).save('new_uv.png')
# img.save('uv_mask.png')