from PIL import Image

# List of image file names in the order you want them in the GIF
fp = "gif1/"
psis = ["01","02", "03","04", "05","06","07", "08", "09", "1", "2", "10"]
fnames = [fp+"psi_" + psi + ".png" for psi in psis]

# Create a list to store the image objects
images = []

# Open and append each image to the list
for filename in fnames:
    img = Image.open(filename)
    images.append(img)

# Save the GIF
images[0].save("output.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
