import numpy as np 
import random
import time
import torch
import torch.utils
import torch.utils.data
import torch.nn as nn


import data
import flow
import pathlib
import argparse


def add_args(parser):
    parser.add_argument("--latent_sie",type=int,default=128)
    parser.add_argument("--variational",choices=["flow","mean-field"])
    parser.add_argument("--flow_depth",type=int,default=2)
    parser.add_argument("--data_size",type=int,default=784)
    parser.add_argument("--learning_rate",type=int,default=0.001)
    parser.add_argument("--max_iteration",type=int,default=30000)
    parser.add_argument("--log_interval",type=int,default=10000)
    parser.add_argument("--n_samples",type=int,default=1000)
    parser.add_argument("--use_gpu",action="store_true")
    parser.add_argument("--seed",type=int,default=582838)
    parser.add_argument("--train_dir",type=pathlib.Path,default="/tmp")
    parser.add_argument("--data_dir",type=pathlib.Path,default="/tmp")
    parser.add_argument("--test_batch_size",type=int,default=512)

    

class Model(nn.Model):
    "Varitional autoencoder, parameterized by a generative network"

    def __init__(self,latent_size,data_size):
        super().__init__()
        # defining the prior distribution p(z) parameters
        self.register_buffer("p_z_loc",torch.zeros(latent_size))  #buffers that are not updated during training and included when saving/loading model
        self.register_buffer("p_z_scales",torch.ones(latent_size))
        self.log_p_z=NormalLogProb() #assumed to be gaussian 
        self.log_p_x=BernoulliLogProb() #assumed to be Bernoulli 
        self.generative_network=NeuralNetwork(
            input_size=latent_size,output_size=data_size,hidden_size=latent_size*2
        )


        def forward(self,z,x):
            "Return log probablity of model for the given latent variable z and observed data x"

            log_p_z=self.log_p_z(self.p_z_loc,self.p_z_scale,z).sum(-1,keepdim=True) #log probability of the prior
            logits=self.generative_network(z) #generative network that maps latent variable z to the parameters of p(x|z)

            # Unsqueeze sample dimension
            logits,x=torch.broadcast_tensors(logits,x.unsqueeze(1))

            log_p_x=-self.log_p_x(logits,x).sum(-1,keepdim=True) #log probablity of the Likelihood p(x|z) whose parameter are modelled by the neural network
            return log_p_z+log_p_x #models the joint log-probablity p(z,x) 


class VaritionalMeanField(nn.Module):
    "Approximate posterior distribution q(z|x) parameterized by an inference network(a nn)"


    def __init__(self,latent_size,data_size):
        super().__init__()
        self.inference_network=NeuralNetwork(
            input_size=data_size,
            output_size=latent_size*2,
            hidden_size=latent_size*2
        )
        self.log_q_z=NormalLogProb()

        self.softplus=nn.Softplus() #ensure outputs strictly positive



    def forward(self,x,n_samples=1):  #n_samples to taken out from approximate distribution

        "Returns sample of latent variable and log prob of the sample under q(z|x) "
        
        # computing parameters of the approximate posterior distribution q(z|x)
        loc,scale_arg=torch.chunk(
            self.inference_network(x).unsqueeze(1),chunks=2,dim=1
        )
        scale=self.softplus(scale_arg)
        eps=torch.randn((loc.shape[0],n_samples,loc.shape[-1]),device=loc.device)
        
        z=loc+scale*eps # Reparamaterization Trick

        log_q_z=self.log_q_z(loc,scale,z).sum(-1,keepdim=True)

        return z,log_q_z


class VaritionalMeanField(nn.Module):

    "Approximate posterior parameterized by flow(https://arxiv.org/abs/1606.04934)"

    def __init__(self,latent_size,data_size,flow_depth):
        super().__init__()
        hidden_size=latent_size*2
        self.inference_network=NeuralNetwork(
            input_size=data_size,
            output_size=latent_size*3,
            hidden_size=hidden_size
        )
        modules=[]

        for _ in range(flow_depth):
            modules.append(
                flow.InverseAutoregressiveFlow(
                    num_input=latent_size,
                    num_hidden=hidden_size,
                    num_context=latent_size
                )
            )

            modules.append(flow.Reverse(latent_size))

        self.q_z_flow=flow.FlowSequential(*modules)
        self.log_q_z_0=NormalLogProb()
        self.softPlus=nn.SoftPlus()


    def forward(self,x,n_samples=1):
        "Return sample of latent variable and log prob of the sample under q(z|x)"
        loc,scale_arg,h=torch.chunk(
            self.inference_network(x).unsqueeze(1),chunks=3,dim=1
        )
        scale=self.softplus(scale_arg)
        eps=torch.randn((loc.shape[0],n_samples,loc.shape[-1]),device=loc.device)
        z_0=loc+scale*eps
        
        log_q_z_0=self.log_q_z_0(loc,scale,z_0)

        z_T,log_q_z_flow=self.q_z_flow(z_0,context=h)

        log_q_z=(log_q_z_0+log_q_z_flow).sum(-1,keepdim=True)
        return z_T,log_q_z
    


class NeuralNetwork(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
        super().__init__()
        modules=[
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size)
        ]

        self.net=nn.Sequential(*modules)
    
    def forward(self,input):
        return self.net(input)



class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,loc,scale,z):
        var=torch.pow(scale,2)  
        return -0.5*torch(2*np.pi*var)-torch.pow(z-loc,2)/(2*var)
    


class BernoulliLogProb(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits=nn.BCEWithLogitsLoss(reduction="none")

    
    def forward(self,logits,target):
        # bernoulli log prob is equivallent to negative binary cross entropy
        return self.bce_with_logits(logits,target)
    




def cycle(iterable):
    while True:
        for x in iterable:
            yield x


@torch.no_grad()
def evaluate(n_samples,model,variational,eval_data):
    model.eval()
    total_log_p_x=0.0
    total_elbo=0.0
    for batch in eval_data:
        x=batch[0].to(next(model.parameters()).device)
        z,log_q_z=variational(x,n_samples)
        log_p_x_and_z=model(z,x)

        elbo=log_p_x_and_z-log_q_z
        log_p_x=torch.logsumexp(elbo,dim=1)-np.log(n_samples)

        total_elbo+=elbo.cpu().numpy().sum()

        total_log_p_x+=log_p_x.cpu().numpy().sum()

    n_data=len(eval_data.dataset)
    return total_elbo/n_data,total_log_p_x/n_data




if __name__== "__main__":
    start_time=time.time()
    parser=argparse.ArgumentParser()
    add_args(parser)
    cfg=parser.parse_args()

    device=torch.device("cuda:0" if cfg.use_gpu else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

