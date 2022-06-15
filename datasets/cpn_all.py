import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
#from .splits import split_dataset
'''
if __package__ is None:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from splits import split_dataset
else:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from .splits import split_dataset
'''


class CPNall(data.Dataset):
    """
    Args:6
        root (string): Root directory of the VOC Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, datatype = 'CPN_all', image_set = 'train', 
                    transform = None, is_rgb = True, kfold = 0, kftimes = 0):

        self.root = root
        self.datafolder = datatype
        self.image_set = image_set
        self.transform = transform
        self.is_rgb = is_rgb
        self.kfold = kfold
        self.kftimes = kftimes

        cpn_root = os.path.join(self.root, 'CPN_all')
        image_dir = os.path.join(cpn_root, 'Images')
        mask_dir = os.path.join(cpn_root, 'Masks')

        if not os.path.exists(cpn_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if kfold == 1:
            splits_dir = os.path.join(root, self.datafolder, 'splits')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        elif kfold > 1 and kfold < 11:
            splits_dir = os.path.join(root, self.datafolder, 'splits', 'cv' + str(kfold), str(kftimes))
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        else:
            raise RuntimeError('Error: K-fold cv')

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered!' 
                             'Please use image_set="train" or image_set="val"')
        
        print("Datatype [%s]: " % self.image_set, self.datafolder)
        print("Data file directory: %s" % split_f)

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError
        
        if self.is_rgb:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')         
        else:
            img = Image.open(self.images[index]).convert('L')
            target = Image.open(self.masks[index]).convert('L')            

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from splits import split_dataset

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=0.485, std=0.229)
            ])
    
    dlist = ['CPN_FH', 'CPN_FN', 'CPN_FN+1', 'CPN_FN+2', 'CPN_FN+3', 'CPN_FN+4']
    dlist = ['CPN_six']
    for j in dlist:
            
        dst = CPNall(root='/data1/sdi/datasets', datatype=j, image_set='val',
                                    transform=transform, is_rgb=True, kfold=5, kftimes=3)
        train_loader = DataLoader(dst, batch_size=5,
                                    shuffle=True, num_workers=2, drop_last=True)
        print('set: %d' % (len(dst)))
        for i, (ims, lbls) in tqdm(enumerate(train_loader)):
            print(ims.shape)
            print(lbls.shape)
            print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            if i > 1:
                break
        