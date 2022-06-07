import os
import sys
import torch.utils.data as data
import numpy as np
from collections import namedtuple
from PIL import Image


class CPNcm(data.Dataset):
    """
    Args:6
        root (string): Root directory of the ``CPN`` and ``Median`` Dataset.
        datatype (string): Dataset type (default: ``CPN_cm``)
        image_set (string): Select the image_set to use, ``train`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        is_rgb (bool): Decide input 3-channel for ``True`` 1-Channel for ``False``
    """
    
    CpnSixClass = namedtuple('CpnSixClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CpnSixClass('background', 0, 0, 'void', 0, False, True, (0, 0, 0)),
        CpnSixClass('nerve', 1, 1, 'void', 0, False, True, (0, 0, 255))
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, root, datatype='CPN_cm', image_set='train', transform=None, is_rgb=True):
        
        is_aug = True

        self.root = os.path.expanduser(root)
        self.image_set = image_set
        self.transform = transform
        self.is_rgb = is_rgb

        cpn_root = os.path.join(self.root, 'CPN_six')
        median_root = os.path.join(self.root, 'Median')
        cpn_image_dir = os.path.join(self.root, 'CPN_all/Images')
        median_image_dir = os.path.join(self.root, 'Median/Images')
        cpn_mask_dir = os.path.join(self.root, 'CPN_all/Masks')
        median_mask_dir = os.path.join(self.root, 'Median/Masks')

        if not os.path.exists(cpn_root) or not os.path.exists(median_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        cpn_splits_dir = os.path.join(cpn_root, 'splits')
        cpn_split_f = os.path.join(cpn_splits_dir, image_set.rstrip('\n') + '.txt')
        median_split_dir = os.path.join(median_root, 'splits')
        median_split_f = os.path.join(median_split_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(cpn_split_f) or not os.path.exists(median_split_f):
            raise ValueError('Wrong image_set entered!' 
                             'Please use image_set="train" or image_set="val"')

        with open(os.path.join(cpn_split_f), "r") as f:
            cpn_file_names = [x.strip() for x in f.readlines()]
        with open(os.path.join(median_split_f), "r") as f:
            median_file_names = [x.strip() for x in f.readlines()]

        if is_aug and image_set=='train':
            self.images = [os.path.join(cpn_image_dir, x + ".bmp") for x in cpn_file_names]
            self.masks = [os.path.join(cpn_mask_dir, x + "_mask.bmp") for x in cpn_file_names]
            self.median_images = [os.path.join(median_image_dir, x + ".jpg") for x in median_file_names]
            self.median_masks = [os.path.join(median_mask_dir, x + ".jpg") for x in median_file_names]
            self.images.extend(self.median_images)
            self.masks.extend(self.median_masks)
        else:
            self.images = [os.path.join(cpn_image_dir, x + ".bmp") for x in cpn_file_names]
            self.masks = [os.path.join(cpn_mask_dir, x + "_mask.bmp") for x in cpn_file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError("Error: ", self.images[index])
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError("Error: ", self.masks[index])
        
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

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from splits import split_dataset

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.GaussianBlur(kernel_size=(5, 5)),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=0.485, std=0.229)
            ])
    
    dlist = ['CPN_cm']

    for j in dlist:
            
        dst = CPNcm(root='/mnt/server5/sdi/datasets', datatype=j, image_set='train',
                                    transform=transform, is_rgb=True)
        train_loader = DataLoader(dst, batch_size=1,
                                    shuffle=True, num_workers=2, drop_last=True)
        print("Train set: %d" % len(train_loader))
        
        for i, (ims, lbls) in tqdm(enumerate(train_loader)):
            print(ims.shape)
            print(lbls.shape)
            print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            if i > 1:
                break
        