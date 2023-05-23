import random
import torch
import numpy as np
import torchvision
import torchvision.transforms.functional as TF

class RandomRotation90(object):
    '''
        Apply random vertical/horizontal flip 
        
        Taken from augment_rot90 from MIC-DKFZ/batchgenerators
        https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/augmentations/spatial_transformations.py
    '''

    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot

    def __call__(self, sample):
        '''
        '''
        num_rot = np.random.choice(self.num_rot)      
        
        def f(img, axes):
            return (np.ascontiguousarray(np.rot90(img, num_rot, axes)))
        
        data, label = sample
        new_sample = torch.from_numpy(f(data, (1,2))), torch.from_numpy(f(label, (0,1)))
        return new_sample

class RandomAffineTransform:
    """ Apply random affine transform : rotation and/or translation and/or scale"""
    
    def __init__(self, degrees=(-180, 180), translate=(0.4, 0.4), scale=(0.5, 1.5)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale

    def __call__(self, sample):
        randomtransform = torchvision.transforms.RandomAffine(self.degrees, self.translate, self.scale)
        data, label = sample
        # apply the transform on all data simultaneously 
        new_sample = randomtransform(torch.concat([data, label.unsqueeze(0)], axis=0))
        return new_sample[:-1], new_sample[-1].squeeze(0)
    
class RandomCrop:
    """ """
    def __init__(self, size=(128,128)):
        self.size = size

    def __call__(self, sample):
        randomtransform = torchvision.transforms.RandomCrop(self.size)
        data, label = sample
        # apply the transform on all data simultaneously 
        new_sample = randomtransform(torch.concat([data, label.unsqueeze(0)], axis=0))
        return new_sample[:-1], new_sample[-1].squeeze(0)

class GaussianBlur:
    """ """
    def __init__(self, kernel_size=(5, 9), sigma=(0.1, 5)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, sample):
        transform = torchvision.transforms.GaussianBlur(self.kernel_size, self.sigma)
        data, label = sample
        # apply the transform on all data simultaneously 
        new_sample = transform(data)
        return new_sample, label
