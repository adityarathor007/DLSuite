import torch

class LinearNoiseSchedular:
    
    def __init__(self,num_timesteps,beta_start,beta_end):
        self.num_timesteps=num_timesteps
        self.beta_start=beta_start
        self.beta_end=beta_end
        
        self.betas=torch.linspace(beta_start,beta_end,num_timesteps)
        self.alphas=1-self.betas
        self.alpha_cum_prod=torch.cumprod(self.alphas,dim=0)
        self.sqrt_alpha_cum_prod=torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod=torch.sqrt(1-self.alpha_cum_prod)

    
    # forward process of adding noise
    def add_noise(self,original,noise,t):
        original_shape=original.shape
        batch_size=original_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(original.device)[t].reshape(batch_size)

        for _ in range(len(original_shape)-1):
            sqrt_alpha_cum_prod=sqrt_alpha_cum_prod.unsqueeze(-1)  #[batch_size]->[batch_size,1]
            sqrt_one_minus_alpha_cum_prod=sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        # image at timestep t using the forward process equation
        return sqrt_alpha_cum_prod*original+sqrt_one_minus_alpha_cum_prod*noise
    

    # backward process of sampling 
    def sample_prev_timestep(self,xt,noise_pred,t):
        
        # using the same eqation for forward process(x0 to xt) and using the noise prediction instead of the actual noise
        x0=(xt-(self.sqrt_one_minus_alpha_cum_prod*noise_pred))/self.sqrt_alpha_cum_prod

        x0=torch.clamp(x0,-1,1)

        
        mean=xt-((self.betas[t]*noise_pred)/(self.sqrt_one_minus_alpha_cum_prod))
        mean=mean/torch.sqrt(self.alphas[t])

        #at timestep 0 we just return the mean whereas for other timestep noise is added 
        if t==0:
            return mean,x0 
        else:
            variance=(1-self.alphas[t])*(1-self.alpha_cum_prod[t-1])
            variance=variance/1-self.alpha_cum_prod[t]

            sigma=variance**0.5

            #using the reparameterization technique 
            z=torch.randn(xt.shape).to(xt.device) #sampling from a gaussian distribution

            return mean+sigma*z,x0
        



