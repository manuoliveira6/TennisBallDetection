import json
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class KeypointsDataset(Dataset):
    def __init__(self, data_dir, file):
        self.data_dir = data_dir
        with open(file, "r") as f:
            self.data = json.load(f)
        
    def transform_img(self, img):

        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img)

        return img

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, id):
        item = self.data[id]
        img = cv2.imread(f"{self.data_dir}/{item['id']}.png")
        h = img.shape[0]
        w = img.shape[1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform_img(img)

        kps = np.array(item['kps']).flatten().astype(np.float32)
        kps[::2] *= 224.0 / w # Adjust w 
        kps[1::2] *= 224.0 / h # Adjust h 
        
        return img, kps