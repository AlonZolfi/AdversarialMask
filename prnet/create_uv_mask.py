from PIL import Image, ImageDraw

im = Image.new(mode='RGB', size=(256,256))

x, y = im.size
eX, eY = 150, 120 #Size of Bounding Box for ellipse

bbox =  (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2)
draw = ImageDraw.Draw(im)
draw.ellipse(bbox, fill='white')
del draw

im.save("output.png")
im.show()