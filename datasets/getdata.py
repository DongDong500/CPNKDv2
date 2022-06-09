from .cpn import CPNSegmentation
from .cpn_all import CPNALLSegmentation
from .cpn_six import CPN
from .median import Median
from .cpn_aug import CPNaug
from .cpn_ver import CPNver
from .cpn_cm import CPNcm
from utils.ext_transforms import ExtCompose

def cpn(root: str = '/', datatype:str = 'CPN', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Peroneal nerve 
    """
    return CPNSegmentation(root, datatype, image_set, transform, is_rgb)

def cpn_all(root: str = '/', datatype:str = 'CPN_all', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
    
    Args:
        root (str): path to data parent directory (Ex: /data1/sdi/datasets). 
        datatype (str): data folder name (default: CPN_all).
        image_set (str): train/val or test (default: train).
        transform (ExtCompose): composition of transform class.
        is_rgb (bool): 3 input channel for True else False.
    """
    return CPNALLSegmentation(root, datatype, image_set, transform, is_rgb)

def cpn_six(root: str = '/', datatype:str = 'CPN_six', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Peroneal nerve (six parts: FH, FN+0 ~ 4)
        410 samples
    """
    return CPN(root, datatype, image_set, transform, is_rgb)

def cpn_aug(root: str = '/', datatype:str = 'CPN_aug', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Peroneal nerve with augmentation
    """
    return CPNaug(root, datatype, image_set, transform, is_rgb)

def cpn_cm(root: str = '/', datatype:str = 'CPN_cm', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Peroneal nerve and median nerve (cpn & median)
    """
    return CPNcm(root, datatype, image_set, transform, is_rgb)

def median(root: str = '/', datatype:str = 'Median', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    """ Median nerve
        1044 + 261 = 1305 samples
    """
    return Median(root, datatype, image_set, transform, is_rgb)
