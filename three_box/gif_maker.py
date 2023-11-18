from PIL import Image
import numpy as np

# List of image file names in the order you want them in the GIF
fp = "gif3/"
psis = np.linspace(-2,2,20)
fnames = [fp+f"pp_{i}.png" for i in range(20)]

# Create a list to store the image objects
images = []

# Open and append each image to the list
for filename in fnames:
    img = Image.open(filename)
    images.append(img)

# Save the GIF
images[0].save("gif3/three_box.gif", save_all=True, append_images=images[1:], duration=175, loop=0)
