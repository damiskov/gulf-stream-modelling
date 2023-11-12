from PIL import Image
import numpy as np

# List of image file names in the order you want them in the GIF
fp = "gif2/"
psis = np.linspace(-2,2,20)
fnames = [fp+f"{i}.png" for i in psis]

# Create a list to store the image objects
images = []

# Open and append each image to the list
for filename in fnames:
    img = Image.open(filename)
    images.append(img)

# Save the GIF
images[0].save("psi_2D.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
