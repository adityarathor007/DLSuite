# 1. Design model (input and output size, forward pass)
# 2. Construct loss and optimizer
# 3. Training Loop:
#        - forward pass: compute prediction
#        - backward pass: gradients
#        - update weights

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# X= torch.tensor([1,2,3,4],dtype=torch.float32)
# y= torch.tensor([2,4,6,8],dtype=torch.float32)
# w= torch.tensor(0.0,dtype=torch.float32,requires_grad=True)
# # model prediction
# def forward(x):
#     return w*x



# modifying the inputs as to use the torch training module
X=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y= torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)


n_samples,n_features=X.shape
print(n_samples,n_features)


# using the torch linear module instead a forward function for prediction
input_size=n_features
output_size=n_features

# model=nn.Linear(input_size,output_size)  #but what if we have multiple layers
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression,self).__init__()
        # define layers
        self.lin=nn.Linear(input_size,output_size)
    
    def forward(self,x):
        return self.lin(x)

model=LinearRegression(input_size,output_size)



# print(f'Prediction before training: f(5)={forward(5):.3f}')
X_test=torch.tensor([5],dtype=torch.float32)
print(f'Predciton before training: f(5)= {model(X_test).item():.3f}')



learning_rate=0.01
n_iters=100
loss=nn.MSELoss()
# optimizer=torch.optim.SGD([w],lr=learning_rate)
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)


for epoch in range(n_iters):
    
    # forward pass -> gradient
    # y_pred=forward(X)
    y_pred=model(X)

    # loss
    l=loss(y,y_pred)

    # backward_pass->gradient
    l.backward() #dl/dw

    # update_weight
    # with torch.no_grad():
    #     w-=learning_rate*w.grad
    # because we are using torch.optim then we can do the weight update using that 
    optimizer.step()
    
    # zero gradients
    # w.grad_zero()
    optimizer.zero_grad()  #By default, PyTorch accumulates gradients. This means that every time loss.backward() is called, the gradients are added to the grad attribute of the parameters, rather than replacing them.
    #It ensure that the gradients calculated during the backward() pass are only based on the current batch of data loss not of some previous batch

    if epoch%10==0:
        [w,b]=model.parameters() #w will also be in 2d array as inputs and each w will have single value in [] which is thus at index zero
        print(f'epoch {epoch+1}: w={w[0][0].item():.3f},loss={l:.8f}')

    
# print(f'Prediction after training: f(5)={forward(5):.3f}')
print(f'Prediction after training: f(5)= {model(X_test).item():.3f}')



# plot
predicted=model(X).detach().numpy()
plt.plot(X,y,'ro')
plt.plot(X,predicted,'b')
plt.show()



