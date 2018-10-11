from PIL import Image
from PIL import ImageOps
import pyscreenshot as ImageGrab #screengrab on Win/*nix, pillow won't do Linux
import pytesseract
import numpy as np
import os

#take screenshot and convert to grayscale
#im = ImageGrab.grab(170, 670, 614, 960)

#using screenshot for now, easier to test
#crop values are the area of the screenshot where the text is (for now)
#will have to be dynamic at some point due to different resolutions, 16x9, 4x3

im = Image.open("./img/csgo-1.png").crop((170, 670, 614, 960))
data = np.array(im)
red, green, blue, alpha = data.T

#take brighter px of image and convert dark pixels to black for better contrast
dark_areas = (red <= 155) & (blue <= 155) & (green <= 155)
data[..., :-1][dark_areas.T] = (0,0,0) #make dark areas black
#gray = im.convert('1')
#gray.show()

im2 = Image.fromarray(data)
im2.show()
text = pytesseract.image_to_string(im2)
print(text)
