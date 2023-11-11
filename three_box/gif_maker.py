from PIL import Image

# List of image file names in the order you want them in the GIF
fp = "gif1/"
gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, 0.8, 0.9, 1]
fnames = [fp+f"gamma_2D_{i}.png" for i in gammas]

# Create a list to store the image objects
images = []

# Open and append each image to the list
for filename in fnames:
    img = Image.open(filename)
    images.append(img)

# Save the GIF
images[0].save("gamma_2D.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
