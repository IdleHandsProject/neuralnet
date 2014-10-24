from PIL import Image, ImageFilter
from PIL import ImageEnhance


image = Image.open('traindata.jpg')
bw = image.convert('L')
th = 100
##bright = ImageEnhance.Brightness(bw)
##bw = bright.enhance(0.1)

##cont = ImageEnhance.Contrast(bw)
##bw = cont.enhance(50)
##image 
bw = bw.point(lambda i: i < th and 255)

bw = bw.filter(ImageFilter.FIND_EDGES)




bw.save('outline.jpg') 
