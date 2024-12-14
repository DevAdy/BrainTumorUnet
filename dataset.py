import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        img_filename = self.images[index]
        img_path = os.path.join(self.image_dir, img_filename)

        mask_filename = img_filename
        mask_path = os.path.join(self.mask_dir,mask_filename)

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        image = image[np.newaxis, :, :]

        mask = np.array(Image.open(mask_path).convert("L"),dtype=np.float32)
        mask[mask > 0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image,mask  


