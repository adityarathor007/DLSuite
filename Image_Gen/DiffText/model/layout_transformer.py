
from transformers import CLIPTokenizer,CLIPTextModel
import re
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt






class TextConditioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer=CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14') #processes the tokenized text to extract embeddings
        self.tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14') #convert raw text into tokens(mapping of words)

        
        self.transformer.eval()

        for param in self.transformer.parameters():
            param.requires_grad=False


    def forward(self, prompt):
        print("The prompt is: ",prompt)
        batch_encoding = self.tokenizer(prompt, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # print("output from tokenizer:",batch_encoding)
        
        # words = self.tokenizer.convert_ids_to_tokens(batch_encoding['input_ids'].view(-1).tolist()) 
        # print(words)

        text_embedding = self.transformer(batch_encoding["input_ids"].cuda())
        # print(text_embedding.last_hidden_state.shape)
        return text_embedding.last_hidden_state.cuda(), batch_encoding["attention_mask"].cuda() # 1, 77, 768 and  1, 768

class LayoutTransformer(nn.Module):
    def __init__(self,layer_number=2):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = torch.nn.TransformerEncoder(
            self.encoder_layer, num_layers=layer_number 
        )
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.decoder_transformer = torch.nn.TransformerDecoder(
            self.decoder_layer, num_layers=layer_number 
        )

        self.mask_embedding = nn.Embedding(2,512)
        self.length_embedding = nn.Embedding(256,512)
        self.width_embedding = nn.Embedding(256,512)
        self.position_embedding = nn.Embedding(256,512)
        self.state_embedding = nn.Embedding(256,512)
        self.match_embedding = nn.Embedding(256,512)
        
        self.x_embedding = nn.Embedding(512,512)
        self.y_embedding = nn.Embedding(512,512)
        self.w_embedding = nn.Embedding(512,512)
        self.h_embedding = nn.Embedding(512,512)
        
        self.encoder_target_embedding = nn.Embedding(256,512)

        self.input_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.output_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        
    def forward(self,x,length,width,mask,state,match,target,right_shifted_boxes,train=False,encoder_embedding=None):
        """
        parameters:
            - x: text_embedding(1,77,768)
            - length: length_list
            - width: width_list (1,77)
            - mask:attention mask(relevant content)
            - state: list stores 0: start, 1: middle, 2: special token(sot and eot) extended to 77
            - match: list storing index of the word in the caption extended to 77
            - target: store state and box coordinates of each token 
            - right_shifted_boxes: 2d array storing [0,0,0,0] for each keyword along extended to 8 size


        """
        # detect whether the encoder_embedding is cached
        if encoder_embedding is None:
            # augmentation
            if train:
                width=width+torch.randint(-3,3,(width.shape[0],width.shape[1])).cuda()


            x=self.input_layer(x)
            
            width_embedding=self.width_embedding(torch.clamp(width,0,255).long()) #(1,77,512)
            encoder_target_embedding=self.encoder_target_embedding(target[:,:,0].long()) #(1,77,512) encode the state of the token state
            pe_embedding=self.position_embedding(torch.arange(77).cuda().unsqueeze(0)) #(1,77,512)
            total_embedding=x+width_embedding+encoder_target_embedding+pe_embedding
            total_embedding=total_embedding.permute(1,0,2) #(77,1,512)
            encoder_embedding=self.transformer(total_embedding) #(77,1,512)


        right_shifted_boxes_resize=(right_shifted_boxes*512).long() #(1,8,4)
        right_shifted_boxes_resize=torch.clamp(right_shifted_boxes_resize,0,512) #(1,8,4)



        # decoder_pe
        pe_decoder=torch.arange(8).cuda()
        pe_embedding_decoder=self.position_embedding(pe_decoder).unsqueeze(0) #(1,8,512)
        decoder_input=pe_embedding_decoder+self.x_embedding(right_shifted_boxes_resize[:,:,0])+self.y_embedding(right_shifted_boxes_resize[:,:,1])+self.w_embedding(right_shifted_boxes_resize[:,:,2])+self.h_embedding(right_shifted_boxes_resize[:,:,3]) #(1,8,512)

        decoder_input=decoder_input.permute(1,0,2) #(8,1,512)

        # generate triangular mask
        mask=nn.Transformer.generate_square_subsequent_mask(8) #(8,8)  #attend to prior tokens in the sequence
        mask=mask.cuda() #(8,8)
        print(decoder_input.shape)
        print(encoder_embedding.shape)
        print("Mask: ",mask)
        decoder_result = self.decoder_transformer(decoder_input, encoder_embedding, tgt_mask=mask) # (8, 1, 512)
        print(decoder_result.shape)
        decoder_result=decoder_result.permute(1,0,2) #(1,8,512)

        box_prediction=self.output_layer(decoder_result)
        print("predicted box size: ",box_prediction)
        
        return box_prediction,encoder_embedding                