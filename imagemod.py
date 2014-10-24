from PIL import Image
image_file = Image.open("Original.jpg") # open colour image
image_file = image_file.convert('1') # convert image to black and white
image_file.save('blackandwhite.jpg')
