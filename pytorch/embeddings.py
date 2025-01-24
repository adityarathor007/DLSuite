import torch
import torch.nn as nn

#allow to insert vector in place of word indexes 


n_embeddings,dim=10,4
emb_1=nn.Embedding(n_embeddings,dim)  

emb_1.weight

inp=torch.LongTensor([[1,3],[5,5]])
emb_1(inp) #querying the rows with the underlying weight array


emb_2=nn.Embedding(n_embeddings,dim,padding_idx=5)
emb_2.weight

print(emb_2(inp))
emb_2(inp).mean().backward() #it will be just 1/16 as 16 terms during mean computation df/d(x1)=1/16
emb_2.weight.grad