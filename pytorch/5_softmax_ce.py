import torch 
import torch.nn as nn
import numpy as np

# calculating softmax using function
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x),axis=0)



# x=np.array([2.0,1.0,0.1])
# outputs=softmax(x)
# print('softmax_numpy: ', outputs)

# # method-2(using torch)
# x=torch.tensor([2.0,1.0,0.1])
# outputs=torch.softmax(x,dim=0)
# print('softmax_torch',outputs)



# calculating cross_entropy using function
# def cross_entropy(actual,predicted):
#     loss= -np.sum(actual*np.log(predicted))
#     return loss


# y must be one encoded
# if class 0 then [1,0,0]

# Y=np.array([1,0,0])

# Y_pred_good=np.array([0.7,0.2,0.1])
# Y_pred_bad=np.array([0.1,0.3,0.6])

# l1=cross_entropy(Y,Y_pred_good)
# l2=cross_entropy(Y,Y_pred_bad)

# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')


# using torch(it itself applies the softmax layer so we dont need to implement sofmax in the last layr)
# nn.LogSoftmax+nn.NLLLoss
# Y has class labels NOT ONE HOT ENOCDED and Y_pred jas raw scores (logits)

loss=nn.CrossEntropyLoss()

# for one sample
# Y=torch.tensor([0]) #just repreosenting the label not one hot encoded
# Y_pred_good=torch.tensor([[2.0,1.0,0.1]])
# Y_pred_bad=torch.tensor([[0.5,2.0,0.3]])

# l1=loss(Y_pred_good,Y)
# l2=loss(Y_pred_bad,Y)

# # output  will be tensor so to get the value
# print(l1.item())
# print(l2.item())


# _,predictions1=torch.max(Y_pred_good,1)
# _,predictions2=torch.max(Y_pred_bad,1)

# print(predictions1)
# print(predictions2)



# eg for 3 samples
Y=torch.tensor([2,1,0])
Y_pred_good=torch.tensor([[0.1,1.0,2.0],[1.0,2.0,0.1],[2.0,1.0,0.1]])
Y_pred_bad=torch.tensor([[2.1,1.0,0.1],[0.1,1.0,2.1],[0.1,3.0,0.1]])


l1=loss(Y_pred_good,Y)
l2=loss(Y_pred_bad,Y)

# output  will be tensor so to get the value
print(l1.item())
print(l2.item())


_,predictions1=torch.max(Y_pred_good,1)
_,predictions2=torch.max(Y_pred_bad,1)

print(predictions1)






# Multiclass_problem

# class NeuralNet1(nn.Module):
#     def __init__(self,input_size,hidden_size,num_classes):
#         super(NeuralNet2,self).init__()
#         self.linear1=nn.Linear(input_size,hidden_size)
#         self.relu=nn.ReLU()
#         self.linear2=nn.Linear(hidden_size,num_classes)
    
#     def forward(self,x):
#         out=self.linear1(x)
#         out=self.relu(out)
#         out=self.linear2(out)

#         # no softmax at the end as using cross entropy loss

#         return out

# model=NeuralNet1(input_size=28*28,hidden_size=5,num_classes=3)
# criterion=nn.CrossEntropyLoss() #(applies softmax)

# Binary Classification

class NeuralNet2(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(NeuralNet2,self).__init__()
        self.linear1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.linear2=nn.Linear(hidden_size,1)
    
    def forward(self,x):
        out=self.linear1(x)
        out=self.relu(out)
        out=self.linear2(out)

        y_pred=torch.sigmoid(out)
        return y_pred


model=NeuralNet2(input_size=28*28,hidden_size=5)
criterion=nn.BCELoss()
