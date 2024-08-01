from PIL import Image


if __name__ == "__main__":
    img = Image.open("picture.jpg")  # Can be many different formats.
    rgb = img.load()
    img_shape = img.size
    print(img_shape)
    print(rgb[0, 0])
