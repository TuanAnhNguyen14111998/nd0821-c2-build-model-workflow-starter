import sys
sys.path.append('..')
from TedAI.tedai import *
from functools import partial
import albumentations as albu

def train_transforms_func(x, img_size):
    albu_tfs = albu.Compose([
        albu.RandomBrightnessContrast(0.4, 0.4),
        albu.ShiftScaleRotate(0.05, (-0.08, 0.01), 20, always_apply=True),
        albu.OneOf([
            albu.SmallestMaxSize(img_size, cv2.INTER_LINEAR),
            albu.SmallestMaxSize(img_size, cv2.INTER_AREA)
        ], p=1.0),
        albu.RandomCrop(img_size, img_size),
        albu.Perspective(0.1, p=0.4),
        albu.HorizontalFlip(p=0.5),   
    ])
    torch_tfs = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        Cutout(10, img_size//20),
    ])

    x = albu_tfs(image=x)['image']
    x = torch_tfs(x)
    return x

class train_transforms:
    def __init__(self, img_size): self.aug_func = partial(train_transforms_func, img_size=img_size)
    def __call__(self, x): return self.aug_func(x)

valid_transforms = lambda img_size: Compose([ToPILImage(), Resize(int(img_size*1.05)), 
                                             CenterCrop((img_size, img_size)),
                                             ToTensor(), 
                                             Normalize(mean=[0.485, 0.456, 0.406], 
                                                       std=[0.229, 0.224, 0.225])])