import os
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

#Call CPU or GPU, GPU has priority to call GPU
print('====================train_nucleus======================')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Weight file from training
weight_path= 'params/Pyramid.pth'
#Import Training Set
data_path=r'DateSet'
#Save training process results
train_save_path= 'train_image_nucleus'
if __name__ == '__main__':

    data_loader=DataLoader(MyDataset(data_path),batch_size=12,shuffle=True)

    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')
    #Determine the learning rate
    opt=optim.Adam(net.parameters(),lr=0.0001)

    loss_fun=nn.BCELoss()

    epoch=1
    while epoch < 2001:

        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)
            out_image=net(image)
            train_loss=loss_fun(out_image,segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            if i % 5 == 0:
                lr = opt.param_groups[0]['lr']
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}', f"Current learning rate: {lr:.6f}")
            #Save weights every 50 epoch
            if epoch%50 == 0:
                torch.save(net.state_dict(),weight_path)
            #It is convenient to observe the effect of each training. Stitching the original image, label image and result image of the 0th image together to see the effect.
            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]
            #Combine these three pictures
            img=torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{train_save_path}/{i}.png')
        epoch+=1

