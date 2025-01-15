import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time 
import os
import copy


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
}

data_dir='data/hymenoptera_data'

image_datasets={x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','val']}


dataloaders={x:torch.utils.data.DataLoader(image_datasets[x],batch_size=4,shuffle=True)  for x in ['train','val']}

dataset_sizes={x:len(image_datasets[x]) for x in ['train','val']}

class_names=image_datasets['train'].classes
print(class_names)

def imshow(inp,title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()



# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since=time.time()


    best_model_wts=copy.deepcopy(model.state_dict())
    best_acc=0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)

        for phase in ['train','val']:
            if phase=='train':
                model.train()  #set model to training mode
            
            else:
                model.eval() #set model to val mode
            
            running_loss=0.0
            running_corrects=0

            for inputs,labels in dataloaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
            

                with torch.set_grad_enabled(phase=='train'):
                    outputs=model(inputs)
                    _,preds=torch.max(outputs,1)
                    loss=criterion(outputs,labels)


                    # backward + optimize only if in training phase
                    if phase=='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
#                 # stats
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)

            if phase=='train':
                scheduler.step()

            epoch_loss=running_loss/dataset_sizes[phase]
            epoch_acc=running_corrects.double()/dataset_sizes[phase]
                
            print(f'{phase} Loss: {epoch_loss:.4f} Acc:{epoch_acc:.4f}')

#                 # deep copy of the model
            if phase=='val' and epoch_acc>best_acc:
                best_acc=epoch_acc
                best_model_wts=copy.deepcopy(model.state_dict())
                
        print()
    
    time_elapsed=time.time()-since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')


    model.load_state_dict(best_model_wts)
    return model


model=models.resnet18(pretrained=True)
num_ftrs=model.fc.in_features  #number of input features for the last layer

model.fc=nn.Linear(num_ftrs,2) #as we have 2 classes 
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)


# # scheduler (to update the lr)

scheduler=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

# for epoch in range(100):
#     train() #optimizer_step
#     evaluate()
#     scheduler.step()


model=train_model(model,criterion,optimizer,scheduler,num_epochs=2)



# # just modifying the last layer
model=models.resnet18(pretrained=True)

# # this freeze the layers
for param in model.parameters():
    param.requires_grad=False


num_ftrs=model.fc.in_features

model.fc=nn.Linear(num_ftrs,2) #as we have 2 classes
model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)


# # scheduler (to update the lr)

scheduler=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

# # for epoch in range(100):
# #     train() #optimizer_step
# #     evaluate()
# #     scheduler.step()


model=train_model(model,criterion,optimizer,scheduler,num_epochs=2)

                

print("training complete")

