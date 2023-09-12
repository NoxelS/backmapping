import glob
import contextlib

from PIL import Image

# filepaths
fp_in = "data/hist/pred/*.png"  # TODO: make with config
fp_out = "training.gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:

    # lazily load images
    imgs = (stack.enter_context(Image.open(f)) for f in sorted(glob.glob(fp_in)))

    images = list(imgs) # make list of images from generator

    # Sort the images and make sure 002 is before 020
    images.sort(key=lambda x: int(x.filename.split("\\")[-1].split(".")[0].split("_")[-1])) # Only works on windows
    
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    images[0].save(fp=fp_out, format='GIF', append_images=images[1:], save_all=True, duration=20, loop=0)