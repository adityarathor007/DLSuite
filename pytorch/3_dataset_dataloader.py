# data=numpy.loadtxt('wine.csv')

# training loop without mini_batch
# for epoch in range(1000):
#     x,y=data
#     # forward+backward+weight_update

# training_loop with mini_batch

# for epoch in range(1000):

#     for i in range(total_batches):
#         x_batch,y_batch = ...


# -> use Dataset and DataLoader to load wine.csv


import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy=np.loadtxt('wine.csv',delimiter=",",dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]]) #n_samples,1
        self.n_samples=xy.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index],self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
    



dataset=WineDataset()

# checking if it is working
# first_data=dataset[0]
# features,labels=first_data
# print(features,labels)

# Dataloader is USED FOR CREATING BATCHES
dataloader=DataLoader(dataset=dataset,batch_size=4,shuffle=True,num_workers=2)
# 2 subprocess load the data in parallel (loading data means creating batches). The advantage of using this is that is since out machine has multiple cores so then next batches will be ready before the main batch is ready for another batch



dataiter=iter(dataloader)
data=next(dataiter) 
features,labels=data #contains 1 batch
print(features,labels)


# TRAINING_LOOP
num_epochs=2
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)  #eg if there are 18 samples then n_iterations will be 5
print(total_samples,n_iterations)


for epoch in range(num_epochs):
    for i,(inputs,labels) in enumerate(dataloader):
        # forward_backward,update
        
        if (i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
