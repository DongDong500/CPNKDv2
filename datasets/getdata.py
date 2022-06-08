from .cpn import CPNSegmentation
from .cpn_all import CPNALLSegmentation
from .cpn_six import CPN
from .median import Median
from .cpn_aug import CPNaug
from .cpn_ver import CPNver
from .cpn_cm import CPNcm
from utils.ext_transforms import ExtCompose

def cpn(root: str = '/', datatype:str = 'CPN', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):

    return CPNSegmentation(root, datatype, image_set, transform, is_rgb)

def cpn_all(root: str = '/', datatype:str = 'CPN_all', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    
    return CPNALLSegmentation(root, datatype, image_set, transform, is_rgb)

def cpn_six(root: str = '/', datatype:str = 'CPN_six', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    
    return CPN(root, datatype, image_set, transform, is_rgb)

def cpn_aug(root: str = '/', datatype:str = 'CPN_aug', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    
    return CPNaug(root, datatype, image_set, transform, is_rgb)

def cpn_cm(root: str = '/', datatype:str = 'CPN_cm', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    
    return CPNcm(root, datatype, image_set, transform, is_rgb)

def medain(root: str = '/', datatype:str = 'Median', image_set:str = 'train', transform:ExtCompose = None, is_rgb:bool = True):
    
    return Median(root, datatype, image_set, transform, is_rgb)
