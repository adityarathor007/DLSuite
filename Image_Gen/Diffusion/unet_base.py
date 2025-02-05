
import torch
import torch.nn as nn



# we also give the timestep we are at along with the image 
def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb



# UNET architecture (most of the diffusion model follow this but differ based on the specification that is happening inside the block)

# Encoder(extracts features and reduce spatial size)
class DownBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,down_sample,num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(8,in_channels), #8 groups each containing in_channels/8 channels and then normalized 
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )
        self.t_emb_layers=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        )

        self.resenet_conv_second=nn.Sequential(
            nn.GroupNorm(8,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )


        self.attention_norm=nn.GroupNorm(8,out_channels)
        self.attention=nn.MultiheadAttention(out_channels,num_heads,batch_first=True)
        
        self.residual_input_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.down_sample_conv=nn.Conv2d(out_channels,out_channels,kernel_size=4,stride=2,padding=1) if self.down_sample else nn.Identity()

    
    def forward(self,x,t_emb):
        out=x

        # Resnet block 
        resnet_input=out
        out=self.resnet_conv_first(out)
        out=out+self.t_emb_layers(t_emb)[:,:,None,None] #adding output from timestep_proj_layer
        out=self.resenet_conv_second(out)
       
        out=out+self.residual_input_conv(resnet_input)

        # Self-Attention Block
        batch_size,channels,h,w=out.shape
        in_attn=out.reshape(batch_size,channels,h*w)
        in_attn=self.attention_norm(in_attn)
        in_attn=in_attn.transpose(1,2) # to ensure the channels features are the last features

        
        out_attn,_=self.attention(in_attn,in_attn,in_attn)
        out_attn=out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out=out+out_attn


        out=self.down_sample_conv(out)
        return out

# BottleNeck(process compressed feature map at the lowest resolution)
class MidBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,num_heads):
        super().__init__()
        self.resnet_conv_first=nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8,in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)

            ),
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)

            )
        ])

        # print("t_emb_dim: ",t_emb_dim)
        # print("Out channels are: ",out_channels)
        self.t_emb_layers=nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim,out_channels)

            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim,out_channels)

            )
        ])

        self.resnet_conv_second=nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8,out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
            ),

        ])


        self.attention_norm=nn.GroupNorm(8,out_channels)
        self.attention=nn.MultiheadAttention(out_channels,num_heads,batch_first=True)


        self.residual_input_conv=nn.ModuleList([
            nn.Conv2d(in_channels,out_channels,kernel_size=1),
            nn.Conv2d(out_channels,out_channels,kernel_size=1)
        ])

    def forward(self,x,t_emb):
        out=x
        # first resnet block 
        resnet_input = out
 
        out = self.resnet_conv_first[0](out)
       

        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out=self.resnet_conv_second[0](out)

        
        # print("Just before error")
        # print(out.shape)
        # print(resnet_input.shape)
        print(self.residual_input_conv[0](resnet_input).shape)
        out=out+self.residual_input_conv[0](resnet_input)

        # attention block

        batch_size,channels,h,w=out.shape
        in_attn=out.reshape(batch_size,channels,h*w)
        in_attn=self.attention_norm(in_attn)
        in_attn=in_attn.transpose(1,2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn=out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out=out+out_attn

        # second resnet block
        resnet_input=out
        out=self.resnet_conv_first[1](out)
        out=out+self.t_emb_layers[1](t_emb)[:,:,None,None]
        out = self.resnet_conv_second[1](out)
        out=out+self.residual_input_conv[1](resnet_input)

        # print("out shape: ",out.shape)
        return out
    

# Decoder(expands the features back to their original resolution)
class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,t_emb_dim,up_sample,num_heads):
        super().__init__()
        self.up_sample=up_sample
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(8,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )

        self.t_emb_layers=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim,out_channels)
        )

        self.resnet_conv_second=nn.Sequential(
            nn.GroupNorm(8,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        )


        self.attention_norm=nn.GroupNorm(8,out_channels)
        self.attention=nn.MultiheadAttention(out_channels,num_heads,batch_first=True)
        self.residual_input_conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)

        self.up_sample_conv=nn.ConvTranspose2d(in_channels//2,in_channels//2,kernel_size=4,stride=2,padding=1) if self.up_sample else nn.Identity()

    def forward(self,x,out_down,t_emb):
        
        x=self.up_sample_conv(x)
        x=torch.cat([x,out_down],dim=1)

        # Resnet_block
        out=x
        resnet_input=out
        out=self.resnet_conv_first(out)
        out=out+self.t_emb_layers(t_emb)[:,:,None,None]
        out=self.resnet_conv_second(out)
        out=out+self.residual_input_conv(resnet_input)

        # Attention_Block   
        batch_size,channels,h,w=out.shape
        in_attn=out.reshape(batch_size,channels,h*w)
        in_attn=self.attention_norm(in_attn)

        in_attn=in_attn.transpose(1,2)
        out_attn,_=self.attention(in_attn,in_attn,in_attn)
        out_attn=out_attn.transpose(1,2).reshape(batch_size,channels,h,w)
        out=out+out_attn

        return out
        
        

class UNet(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        im_channels = model_config['im_channels']
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.t_emb_dim = model_config['time_emb_dim']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']

        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1


        # Time Embedding block had position embedding followed by linear layer with activation in between (this is different from the timestep layers which we had for each resent block this can only be called once in an entire forward pass at start to get the intial time step represetation)

        self.t_proj=nn.Sequential( #called once in forward pass to get the timestep representation
            nn.Linear(self.t_emb_dim,self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim,self.t_emb_dim)
        )


        self.up_sample=list(reversed(self.down_sample))

        # print(im_channels)
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))


        self.downs=nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownBlock(self.down_channels[i],self.down_channels[i+1],self.t_emb_dim,down_sample=self.down_sample[i],num_heads=4))
        
        self.mids=nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            # print(self.mid_channels[i])
            self.mids.append(MidBlock(self.mid_channels[i],self.mid_channels[i+1],self.t_emb_dim,num_heads=4))
        

        self.ups=nn.ModuleList([])

        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpBlock(self.down_channels[i]*2,self.down_channels[i-1] if i!=0 else 16, self.t_emb_dim,up_sample=self.down_sample[i],num_heads=4))



        self.norm_out=nn.GroupNorm(8,16)
        self.conv_out=nn.Conv2d(16,im_channels,kernel_size=3,padding=1)


    def forward(self,x,t):
        out=self.conv_in(x)
        t_emb=get_time_embedding(t,self.t_emb_dim) #get the time embedding from sinusoidal position embedding
        t_emb=self.t_proj(t_emb)

        down_outs=[]
        for down in self.downs:
            # print(out.shape)
            down_outs.append(out)
            out=down(out,t_emb)


        for mid in self.mids:
            print("forward of unet_shape:",out.shape)
            out=mid(out,t_emb)

        for up in self.ups:
            down_out=down_outs.pop()
            # print(out,down_outs.shape)
            out=up(out,down_out,t_emb)
        
        out=self.norm_out(out)
        out=nn.SiLU()(out)
        out=self.conv_out(out)

        return out