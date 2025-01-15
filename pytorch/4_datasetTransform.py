import torch 
import torchvision 
from torch.utils.data import Dataset,DataLoader
import numpy as np

'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class 

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
'''

# dataset=torchvision.dataset.MNIST(
#     root='./data',transform=torchvision.transform.ToTensor()
# )

class WineDataset(Dataset):

    def __init__(self,transform=None):
        # data loading
        xy=np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.n_samples=xy.shape[0]


        self.x=xy[:,1:]
        self.y=xy[:,[0]] #n_samples,1

        self.transform=transform


        


    def __getitem__(self, index):
        # dataset[0]
        sample=self.x[index],self.y[index]

        if self.transform:
            sample=self.transform(sample)

        return sample

    def __len__(self):
        # len(dataset)
        return self.n_samples


class ToTensor:
    def __call__(self,sample):
        inputs,targets=sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)



class MulTransform:
    def __init__(self,factor):
        self.factor=factor
    
    def __call__(self,sample):
        input,target=sample
        input*=self.factor
        return input,target
    
    


dataset=WineDataset(transform=None)
first_data=dataset[0]
feature,label=first_data
print(feature)
print(type(feature),type(label)) 


composed=torchvision.transforms.Compose([ToTensor(),MulTransform(2)])  #compose 2 transform classes
dataset=WineDataset(transform=composed)

first_data=dataset[0]
feature,label=first_data
print(feature)
print(type(feature),type(label)) 
