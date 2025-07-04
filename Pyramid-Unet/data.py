import os
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.name=os.listdir(os.path.join(path,'labels'))
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name=self.name[index]

        segment_path=os.path.join(self.path,'labels',segment_name)

        image_path=os.path.join(self.path,'Images',segment_name.replace('png','jpg'))

        segment_image=keep_image_size_open(segment_path)

        image=keep_image_size_open(image_path)

        return transform(image),transform(segment_image)

if __name__ == '__main__':
    data=MyDataset('DateSet')
    print(data[0][0].shape)
    print(data[0][1].shape)

