
from transformers import CLIPTokenizer,CLIPTextModel
import re
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageOps
from model.layout_transformer import LayoutTransformer, TextConditioner
from util import get_width, get_key_words, adjust_overlap_box, shrink_box, adjust_font_size, alphabet_dic


model=LayoutTransformer().cuda().eval()
model.load_state_dict(torch.load('textdiffuser-ckpt/layout_transformer.pth'))


text_encoder = TextConditioner().cuda().eval()
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

def process_caption(font_path,caption,keywords):

    """
    Input: 
        - font_path: font_style used
        - caption: prompt
        - keyword: which words to be shown in image(written in caption with '')
    Output:
        - caption: after removing puntuation marks
        - length_list: to store the number of letters in each word with extended to 77 
        - width_list: to store the width in terms number of pixels it will take if written using the font_path style with extended to 77 
        - info_array: stores the box coordinates and state(1 for keyword rest 0) of each token extended to 77
        - words: list containing the output from converting ids(token from tokenizer) to word with removing of </w> tag
        - state_list: stores 0: start, 1: middle, 2: special token(sot and eot) extended to 77
        - word_match_list: index of the word in the caption extended to 77
        - boxes: 2d array storing [0,0,0,0] for each keyword along extended to 8 size
        - boxes_length: number of keyword 

    """
    
    
    # remove punctuations. please remove this statement if you want to paint punctuations
    caption = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039])", " ", caption) 
    

    caption_words=tokenizer([caption],truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    print(caption_words)

    caption_words_ids = caption_words['input_ids'] # (1, 77)  (tokens)
    # print(caption_words_ids)
    length = caption_words['length'] # (1, )
    print(length) #sent_len+2 for sos and eos

    # convert id back to words
    words = tokenizer.convert_ids_to_tokens(caption_words_ids.view(-1).tolist()) 
    # print(words)
    words = [i.replace('</w>', '') for i in words]
    # print(words)
    words_valid = words[:int(length)]  #since padding of max_length so after eot tag still eot till 77 will be added
    print(words_valid) 



    # split the caption into words and convert them to lower case
    caption_split = caption.split() 
    caption_split=[i.lower() for i in caption_split]
    print(caption_split)

    start_dic={}  # get the start index of each word
    state_list=[] # 0: start, 1: middle, 2: special token
    word_match_list=[]  # the index of the word in the caption
    current_caption_index=0
    current_match=''

    for i in range(length):
        
        # use first and last token as special tokens
        if i==0 or i==length-1:
            state_list.append(2) 
            word_match_list.append(127)
            continue
            
        if current_match=='':
            state_list.append(0)
            start_dic[current_caption_index]=i

        else:
            state_list.append(1)

        current_match+=words_valid[i]
        word_match_list.append(current_caption_index)
        if current_match==caption_split[current_caption_index]:
            current_match=''
            current_caption_index+=1

        
    
    print(state_list) 
    print(word_match_list)

    while len(state_list)<77:
        state_list.append(127)
    
    while len(word_match_list)<77:
        word_match_list.append(127)

    # print(state_list) 
    # print(word_match_list)


    length_list=[]
    width_list=[]

    for i in range(len(word_match_list)):
        if word_match_list[i]==127:
            length_list.append(0)
            width_list.append(0)
        else:
            length_list.append(len(caption.split()[word_match_list[i]]))  #storing the length of the word
            width_list.append(get_width(font_path,caption.split()[word_match_list[i]])) #for 

    
    length_list = torch.Tensor(length_list).long() # (77, ) with torch.int64
    width_list = torch.Tensor(width_list).long() # (77, )


    boxes=[]
    duplicate_dict={} #some words may appear more than once
    
    # store the box coordinates and state of each token
    info_array = np.zeros((77,5)) # (77, 5)
    

    for keyword in keywords:
        keyword = keyword.lower()
        if keyword in caption_split:
            if keyword not in duplicate_dict:
                duplicate_dict[keyword] = caption_split.index(keyword) #get the index of the keyword in the sentence
                index = caption_split.index(keyword)
            else:
                if duplicate_dict[keyword]+1 < len(caption_split) and keyword in caption_split[duplicate_dict[keyword]+1:]:
                    index = duplicate_dict[keyword] + caption_split[duplicate_dict[keyword]+1:].index(keyword)
                    duplicate_dict[keyword] = index
                else:
                    continue
                
            index = caption_split.index(keyword) 
            index = start_dic[index] 
            info_array[index][0] = 1 #index denote the position of the keyword in the words list

            box = [0,0,0,0] 
            boxes.append(list(box))
            info_array[index][1:] = box
    
    boxes_length=len(boxes)
    if boxes_length>8:  #if keywords are more than 8
        boxes=boxes[:8]
    while len(boxes)<8:
        boxes.append([0,0,0,0])


    return caption,length_list,width_list,torch.from_numpy(info_array),words,torch.Tensor(state_list).long(),torch.Tensor(word_match_list).long(),torch.Tensor(boxes),boxes_length
    

        


def get_layout_from_prompt(args):

    font_path=args.font_path
    print("the font path is:  ",font_path)
    caption=args.prompt
    text_embedding,mask=text_encoder(caption)   #text_embedding is embedding for tokens, and mask is the attention mask(relevant content)
    print(text_embedding.shape)


    keywords=get_key_words(caption)
    print("The following words to be displayed in image were detected",keywords)

    caption,length_list,width_list,target,words,state_list,word_match_list,boxes,boxes_length=process_caption(font_path,caption,keywords)

    
    print("target_shape: ",target.shape)
    target=target.cuda().unsqueeze(0)
    print("target_shape_after_squezzing: ",target.shape)

    print("width_list_shape: ",width_list.shape)
    # print("width_list: ",width_list)
    # print("length_list: ",length_list)
    # print("target: ",target)
    # print("words:",words)
    # print("state_list: ",state_list)
    # print("word_match_list: ",word_match_list)
    # print("boxes",boxes)
    # print("boxes length",boxes_length)




    width_list=width_list.cuda().unsqueeze(0) #(1,77)
    length_list=length_list.cuda().unsqueeze(0) #(1,77)
    state_list=state_list.cuda().unsqueeze(0) #(1,77)
    word_match_list=word_match_list.cuda().unsqueeze(0) #(1,77)


    padding=torch.zeros(1,1,4).cuda()

    # print(boxes.shape)
    # print("boxes: ",boxes)
    boxes=boxes.unsqueeze(0).cuda() #(1,8,4) it is storing keywords boxes with padding till 8
    right_shifted_boxes=torch.cat([padding,boxes[:,0:-1,:]],1)  
    # print("right_shifted_boxes: ",right_shifted_boxes) #the above operation having no effect

    # inference 
    return_boxes=[]
    with torch.no_grad():
        for box_index in range(boxes_length):

            if box_index==0:
                encoder_embedding=None
            
            output, encoder_embedding = model(text_embedding, length_list, width_list, mask, state_list, word_match_list, target, right_shifted_boxes, train=False, encoder_embedding=encoder_embedding)             
            output=torch.clamp(output,min=0,max=1)
            # add overlap detection 
            output=adjust_overlap_box(output,box_index)


            right_shifted_boxes[:,box_index+1,:]=output[:,box_index ,:]
            xmin,ymin,xmax,ymax=output[0,box_index,:].tolist()
            return_boxes.append([xmin,ymin,xmax,ymax])


    
     # print the location of keywords (bbox)
    print(f'index\tkeyword\tx_min\ty_min\tx_max\ty_max')
    for index, keyword in enumerate(keywords):
        x_min = int(return_boxes[index][0] * 512)
        y_min = int(return_boxes[index][1] * 512)
        x_max = int(return_boxes[index][2] * 512)
        y_max = int(return_boxes[index][3] * 512)
        print(f'{index}\t{keyword}\t{x_min}\t{y_min}\t{x_max}\t{y_max}')


    
    #paint the layout
    render_image=Image.new('RGB',(512,512),(255,255,255))
    draw= ImageDraw.Draw(render_image)
    segmentation_mask = Image.new("L", (512,512), 0)
    segmentation_mask_draw = ImageDraw.Draw(segmentation_mask)


    for index,box in enumerate(return_boxes):
        box = [int(i*512) for i in box]
        xmin, ymin, xmax, ymax = box
        
        width = xmax - xmin
        height = ymax - ymin
        text = keywords[index]

        font_size = adjust_font_size(font_path, width, height, draw, text)
        font = ImageFont.truetype(font_path, font_size)

        # draw.rectangle([xmin, ymin, xmax,ymax], outline=(255,0,0))
        draw.text((xmin, ymin), text, font=font, fill=(0, 0, 0))
            
        boxes = []
        for i, char in enumerate(text):
            
            # paint character-level segmentation masks
            # https://github.com/python-pillow/Pillow/issues/3921
            
            # bottom_1=font.getsize(text[i])[1] ##deprecetaed
            # right,bottom_2=font.getsize(text[:i+1])

            bottom_1 = font.getbbox(text[i])[3]
            l,t,r, bottom_2 = font.getbbox(text[:i+1])
            right=r-l
            bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
            width, height = font.getmask(char).size
            right += xmin
            bottom += ymin
            top = bottom - height
            left = right - width
            
            char_box = (left, top, right, bottom)
            boxes.append(char_box)
            
            char_index = alphabet_dic[char]
            segmentation_mask_draw.rectangle(shrink_box(char_box, scale_factor = 0.9), fill=char_index)

    return render_image,segmentation_mask



            
            


