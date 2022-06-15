from .cpn_all import CPNall
from .cpn_six import CPN
from .cpn_aug import CPNaug
from .cpn_cm import CPNcm
from .median import Median
from utils.ext_transforms import ExtCompose


def cpn_all(root: str = '/', datatype:str = 'CPN_all', image_set:str = 'train', 
                transform:ExtCompose = None, is_rgb:bool = True, kfold:int = 0, kftimes:int = 0):
    """ Peroneal nerve (all parts: fiber head (FH), fibular neuropathy (FN+0 ~ 15), POP+0 ~ 5)
        490 samples
    
    Args:
        root (str): path to data parent directory (Ex: /data1/sdi/datasets). 
        datatype (str): data folder name (default: CPN_all).
        image_set (str): train/val or test (default: train).
        transform (ExtCompose): composition of transform class.
        is_rgb (bool): 3 input channel for True else False.
        kfold (int): k-fold cross validation
        kftimes (int): current iteration of cv
    """
    return CPNall(root, datatype, image_set, transform, is_rgb, kfold, kftimes)

def cpn_six(root: str = '/', datatype:str = 'CPN_six', image_set:str = 'train', 
            transform:ExtCompose = None, is_rgb:bool = True, kfold:int = 0, kftimes:int = 0):
    """ Peroneal nerve (six parts: FH, FN+0 ~ 4)
        410 samples
    """
    if kfold == 0:
        return CPN(root, datatype, image_set, transform, is_rgb)
    else:
        raise CPNall(root, datatype, image_set, transform, is_rgb, kfold, kftimes)

def cpn_aug(root: str = '/', datatype:str = 'CPN_aug', image_set:str = 'train', 
            transform:ExtCompose = None, is_rgb:bool = True, kfold:int = 0, kftimes:int = 0):
    """ Peroneal nerve with augmentation
    """
    if kfold == 0:
        return CPNaug(root, datatype, image_set, transform, is_rgb)
    else:
        raise NotImplementedError

def cpn_cm(root: str = '/', datatype:str = 'CPN_cm', image_set:str = 'train', 
            transform:ExtCompose = None, is_rgb:bool = True, kfold:int = 0, kftimes:int = 0):
    """ Peroneal nerve and median nerve (cpn & median)
    """
    if kfold == 0:
        return CPNcm(root, datatype, image_set, transform, is_rgb)
    else:
        raise NotImplementedError

def median(root: str = '/', datatype:str = 'Median', image_set:str = 'train', 
            transform:ExtCompose = None, is_rgb:bool = True, kfold:int = 0, kftimes:int = 0):
    """ Median nerve
        1044 + 261 = 1305 samples
    """
    if kfold == 0:
        return Median(root, datatype, image_set, transform, is_rgb)
    else:
        raise NotImplementedError