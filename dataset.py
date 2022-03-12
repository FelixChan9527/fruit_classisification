from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

data_dir = "/home/MyServer/data/fruit/train_data"
train_list = os.listdir(data_dir)
label_txt = "/home/MyServer/My_Code/MachineLearning/fruit_classification/label.txt"

# for fruit in train_list:
#     f.write(fruit)
#     f.write('\n')
# f.close()
def read_labels(label_txt):
    f = open(label_txt, 'r')
    lines = f.readlines()
    fruit_names = []
    for fruit_name in lines:
        fruit_name = fruit_name.strip('\n')
        fruit_names.append(fruit_name)  
    f.close()

    return fruit_names

class FruitDataset(Dataset):
    def __init__(self, data_dir, fruit_names):
        self.data_dir = data_dir
        self.fruit_names = fruit_names
        self.imgs, self.labels = self.tagging(data_dir, fruit_names)
        self.transforms = A.Compose([
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),   # 标准化，归一化
            ToTensorV2()        # 转为tensor
        ])
    
    def tagging(self, data_dir, fruit_names):
        fruit_imgs = []
        fruit_labels = []
        for idx, fruit_name in enumerate(fruit_names):
            fruit_dir = os.path.join(data_dir, fruit_name)
            imgs_dir = os.listdir(fruit_dir)
            for img_dir in imgs_dir:
                img_dir = os.path.join(fruit_dir, img_dir)
                img = np.array(Image.open(img_dir).convert("RGB"))
                fruit_imgs.append(img)
                fruit_labels.append(idx)
        
        print("#############数据加载完成#############")
        return fruit_imgs, fruit_labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        augmentations = self.transforms(image=img)
        img = augmentations["image"]
        
        return img, label

