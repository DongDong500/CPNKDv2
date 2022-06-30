import os
import socket
import utils
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils import ext_transforms as et

DATA_DIR = {
    3 : "/mnt/server5/sdi/datasets",
    4 : "/mnt/server5/sdi/datasets",
    5 : "/data1/sdi/datasets"
}

def get_datadir():
    if socket.gethostname() == "server3":
        return DATA_DIR[3]
    elif socket.gethostname() == "server4":
        return DATA_DIR[4]
    elif socket.gethostname() == "server5":
        return DATA_DIR[5]
    else:
        raise NotImplementedError

def get_interval(image, result, mean, sigma):

    if mean - sigma > 0:
        result[(image > mean-sigma) * (image < mean+sigma)] = int(mean)
        get_interval(image, result, mean - 2*sigma, sigma)
    else:
        result[(image > mean-sigma) * (image < mean+sigma)] = 0
    return result

def get_up_interval(image, result, mean, sigma):

    if mean + sigma <= 255:
        result[(image > mean-sigma) * (image < mean+sigma)] = int(mean)
        get_up_interval(image, result, mean + 2*sigma, sigma)
    else:
        result[(image > mean-sigma) * (image < mean+sigma)] = 255
    return result

if __name__ == "__main__":

    datadir = os.path.join(get_datadir(), 'CPN_all/Images')
    maskdir = os.path.join(get_datadir(), 'CPN_all/Masks')

    for fname in os.listdir(datadir):
        
        mname = fname.split('.')[0] + "_mask." + fname.split('.')[-1]
        mask = os.path.join(maskdir, mname)
        image = os.path.join('/data1/sdi/datasets/CPN_all/Images', fname)

        if not os.path.exists(mask) or not os.path.exists(image):
            raise Exception ("File Not Exists", mask, image)
        
        image = Image.open(image).convert("L")
        mask = Image.open(mask).convert("L")

        rimage = np.array(image, dtype=np.uint8)[np.where(np.array(mask) > 0)]

        if rimage.ndim != 1:
            raise Exception ("dimensions are not correct", rimage.shape)

        mu = rimage.mean()
        sigma = rimage.std()

        result = get_interval(np.array(image, dtype=np.uint8), np.zeros_like(np.array(image, dtype=np.uint8)), mu, sigma)
        result = Image.fromarray(get_up_interval(np.array(image, dtype=np.uint8), result, mu, sigma))
        result.save(os.path.join(get_datadir(), 'CPN_all_gmm/1sigma', fname, ))
