import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,n_input_features):
        super(Model,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
    
model=Model(n_input_features=7)
# train your model

for param in model.parameters():
    print(param)


FILE="model1.pth"
# torch.save(model,FILE)  #M1 of saving the lazy method (it serializes the entire model, including weights, optimizer state, and sometimes even the architecture, depending on how it's used.)


# model=torch.load(FILE)   
# model.eval()

# for param in model.parameters():
#     print(param)



# M2 of saving model 
# torch.save(model.state_dict(),FILE)


# loaded_model=Model(n_input_features=7)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()


# for param in loaded_model.parameters():
#     print(param)



learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)
print(optimizer.state_dict)  #it will show the learning rate and the momentum


# during training if we want to save a checkpoint 

checkpoint={
    "epoch":90,
    "model_state":model.state_dict(),
    "optim_state":optimizer.state_dict(),

}

# torch.save(checkpoint,"checkpoint.pth")



# loading the checkpoint and continuing the training
loaded_checkpoint=torch.load("checkpoint.pth")
epoch=loaded_checkpoint["epoch"]


model=Model(n_input_features=7)
optimizer=torch.optim.SGD(model.parameters(),lr=0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])  #over-writting the optimizer 

print(optimizer.state_dict())


# Save on GPU, Load on CPU
device=torch.device("cuda")
model.to(device)
torch.save(model._save_to_state_dict(),PATH)


device=torch.device('cpu')
model=Model(*args,**kwargs)
model.load_state_dict(torch.load(PATH,map_location=device))


# Save On CPU, Load on GPU

torch.save(model.state_dict(),PATH)

device=torch.device("cuda")
model=Model(*args,**kwargs)
model.load_state_dict(torch.load(PATH,map_location="cuda:0"))
model.to(device)




