from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy

pic = Image.open("red-fox.jpg")

pix = numpy.array(pic)

print("depth:", pic.bits, "width:", pic.size[0], "heigth:", pic.size[1])

plt.imshow(pix)

img = pic.resize((227, 227))

pix2 = numpy.array(img)

plt.imshow(pix2)

lilpic = pic.crop((100, 327, 100, 327))
pix3 = numpy.array(lilpic)

#plt.imshow(pix3)


#test = pic.convert('LA')

#pix4 = numpy.array(list(test.getdata(band=0)), float)

#plt.imshow(pix4)

rot = pic.rotate(30, expand=1)

pixi = numpy.array(rot)

#plt.imshow(pixi)


print(pic)



plt.show()

