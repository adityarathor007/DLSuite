import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameter
num_epochs=4
batch_size=4
learning_rate=0.001

# dataset has PILImage images of range [0,1]
# We transform them to Tensors of normalized range [-1,1]
transform=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)


train_dataset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
test_dataset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# plotting some samples
# def imshow(img):
#     img=img/2*0.5
#     npimg=img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()

# # get some random training_images
# dataiter=iter(train_loader)
# images,labels=next(dataiter)


# imshow(torchvision.utils.make_grid(images))

# testing the shape after applying convulation operation
# conv1=nn.Conv2d(3,6,5)
# pool=nn.MaxPool2d(2,2)
# conv2=nn.Conv2d(6,16,5)
# print(images.shape)

# x=conv1(images)
# print(x.shape)
# x=pool(x)
# print(x.shape)
# x=conv2(x)
# print(x.shape)
# x=pool(x)
# print(x.shape)






class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5)
        self.fc1=nn.Linear(16*5*5,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x





model=ConvNet().to(device)

criterion=nn.CrossEntropyLoss()
optimzer=torch.optim.SGD(model.parameters(),lr=learning_rate)

n_total_steps=len(train_loader)  #the number of batches 

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):

        images=images.to(device)
        labels=labels.to(device)


        # forward pass
        outputs=model(images)
        loss=criterion(outputs,labels)

        # backward and optimize
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (i+1)%2000==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}],Loss: {loss.item():.4f}')






print('Finish Training')


with torch.no_grad():
    n_correct=0
    n_samples=0
    n_class_correct=[0 for i in range(10)]
    n_class_samples=[0 for i in range(10)]

    for images,labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        outputs=model(images)

        _,predicted=torch.max(outputs,1)
        n_samples+=labels.size(0)
        n_correct+=(predicted==labels).sum().item()

        for i in range(batch_size):
            label=labels[i]
            pred=predicted[i]
            if (label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1

acc=100.0*n_correct/n_samples
print(f'Accuracy of the network: {acc} %')

for i in range(10):
    acc=100.0*n_class_correct[i]/n_class_samples[i]
    print(f'Accuracy of {classes[i]}: {acc}')
