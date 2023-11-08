from PIL import Image

# List of image file names in the order you want them in the GIF
fp = "gif1/"
k_d = [10000, 5000, 1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.01, 0.001, 0.0001]
fnames = [fp+f"2D_kd_{i}.png" for i in k_d]

# Create a list to store the image objects
images = []

# Open and append each image to the list
for filename in fnames:
    img = Image.open(filename)
    images.append(img)

# Save the GIF
images[0].save("2D_k_d_ratio.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
