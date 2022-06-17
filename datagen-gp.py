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

def genImage(im):
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return (denorm(im.numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
    
if __name__ == "__main__":

    datadir = os.path.join(get_datadir(), 'CPN_all/Images')
    maskdir = os.path.join(get_datadir(), 'CPN_all/Masks')
    dstdir = os.path.join(get_datadir(), 'CPN_ver01/splits')

    imgs = [x for x in os.listdir(datadir)]
    masks = [x for x in os.listdir(datadir)]
    
    assert len(imgs) == len(masks)

    ext_trans = et.ExtCompose([ et.ExtToTensor(), 
                                et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                et.GaussianPerturb(std=0.1) ])
    std = 'std010'

    for i in tqdm(range(len(imgs))):
        fname = os.path.join(datadir, imgs[i])
        mname = os.path.join(datadir, masks[i])

        img = Image.open(fname).convert('RGB')
        ma = Image.open(mname).convert('L')
        
        pic, _ = ext_trans(img, ma)
        
        pic = genImage(pic)

        if not os.path.exists(os.path.join(get_datadir(), 'CPN_all_GP/%s/Images' % std)):
            os.makedirs(os.path.join(get_datadir(), 'CPN_all_GP/%s/Images' % std))
        
        Image.fromarray(pic).save(os.path.join(get_datadir(), 'CPN_all_GP/%s/Images' %std, imgs[i]))