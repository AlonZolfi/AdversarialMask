from PIL import Image, ImageDraw

im = Image.new(mode='L', size=(256, 256))

x, y = im.size
eX, eY = 175, 90 #Size of Bounding Box for ellipse

bbox = (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2)
draw = ImageDraw.Draw(im)
draw.ellipse(bbox, fill='white')

# bbox = (26, 97, 56, 159)
# draw.ellipse(bbox, fill='black')
# bbox = (198, 97, 228, 159)
# draw.ellipse(bbox, fill='black')
from torchvision import transforms
img_t = transforms.ToTensor()(im)
img_t[0, :, :50] = 0
img_t[0, :, -50:] = 0

# for i in range(256):
    # print(i, img_t[0, i, :].sum())
# img_t[91:201] = img_t[73:183]
img_t1 = img_t.clone()
# img_t1[0, 91:191] = img_t[0, 78:178]
# img_t1[0, 78:91] = 0
# img_t1[0, 60:90]= img_t[0, 40:70]
# img_t1[0, 40:61] = 0
import torch
im = transforms.ToPILImage()(img_t1).save('../prnet/new_uv.png')

# im.save("output.png")
transforms.ToPILImage()(img_t1).show()

