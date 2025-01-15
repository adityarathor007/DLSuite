# MNIST
# DataLoader,Transformation
# Multilayer Neural Net,activation function
# Loss and Optimizer
# Training Loop(batch training)
# Model evaluation 
# GPU support


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import sys
import torch.nn.functional as F


writer=SummaryWriter("runs/mnist2")

# device config
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# hyper_paramater
input_size=784 #flattening of 28*28
hidden_size=100
num_classes=10
num_epochs=2
batch_size=100
learning_rate=0.01


# MNIST 
train_dataset=torchvision.datasets.MNIST(root='./data',train=True,transform=transform.ToTensor(),download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False,transform=transform.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)

print(f'Number of training batches: {num_train_batches}')
print(f'Number of testing batches: {num_test_batches}')

examples=iter(train_loader)
samples,labels=next(examples)
print(samples.shape,labels.shape)


# # to display samples
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0],cmap='grey')

# plt.show()

img_grid=torchvision.utils.make_grid(samples)
writer.add_image('mnist_images',img_grid)  #to add sample images


class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        return out

model=NeuralNet(input_size,hidden_size,num_classes).to(device)

# loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

writer.add_graph(model,samples.reshape(-1,28*28)) #to add the graph of the architecture

# writer.close()
# sys.exit()



# training_loop
n_total_steps=len(train_loader)

running_loss=0.0
running_correct=0
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        # 100,1,28,28 -> 100,784
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        # forward
        outputs=model(images)
        loss=criterion(outputs,labels)

        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        _,predicted=torch.max(outputs.data,1)
        running_correct+=(predicted==labels).sum().item()

        if(i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training_loss',running_loss/100,epoch*n_total_steps+i)
            writer.add_scalar('training_accuracy',running_correct/100,epoch*n_total_steps+i)
            running_loss=0.0
            running_correct=0




# test


labels1=[]
preds=[]
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images,labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)
        outputs=model(images)

        # value,index
        _,predictions=torch.max(outputs,1)

        n_samples+=labels.shape[0]
        n_correct+=(predictions==labels).sum().item()


        class_prediction=[F.softmax(output,dim=0) for output in outputs]

        preds.append(class_prediction)
        labels1.append(predicted)

    labels=torch.cat(labels1)
    preds=torch.cat([torch.stack(batch) for batch in preds])


    acc=100.0*n_correct/n_samples
    print(f'accuracy={acc}')


    classes=range(10)

    for i in classes:
        labels_i=labels==i
        preds_i=preds[:,i]
        writer.add_pr_curve(str(i),labels_i,preds_i,global_step=0)
        writer.close()

        

        