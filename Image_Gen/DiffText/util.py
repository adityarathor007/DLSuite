import string 

from PIL import Image, ImageFont, ImageDraw, ImageOps



alphabet=string.digits+string.ascii_lowercase+string.ascii_uppercase+string.punctuation+' '
alphabet_dic={}
for index,c in enumerate(alphabet):
    alphabet_dic[c]=index-1 #the index starts with 0 for for non-character


def adjust_font_size(font_path,width,height,draw,text):

    size_start=height
    while True:
        font=ImageFont.truetype(font_path,size_start)
        text_width=draw.textlength(text,font=font)
        if text_width>=width:
            size_start=size_start-1
        else:
            return size_start

def get_width(font_path, text):
    """
    This function calculates the width of the text.
    
    Args:
        font_path (str): user prompt.
        text (str): user prompt.
    """
    font = ImageFont.truetype(font_path, 24)
    # bbox = font.getbbox(text)
    # width=bbox[2]-bbox[0]
    # height=bbox[3]-bbox[1]

    width=font.getlength(text)

    return width

def get_key_words(text: str):
    """
    This function detect keywords (enclosed by quotes) from user prompts. The keywords are used to guide the layout generation.
    
    Args:
        text (str): user prompt.
    """

    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    
    if matches:
        for match in matches:
            # words.append(match.split())
            words.append(match)
            
    if len(words) >= 8:
        return []
    
    # print(words)
    
    return words

def filter_segmentation_mask(segmentation_mask: np.array):
    """
    This function removes some noisy predictions of segmentation masks.
    
    Args:
        segmentation_mask (np.array): The character-level segmentation mask.
    """
    segmentation_mask[segmentation_mask==alphabet_dic['-']] = 0
    segmentation_mask[segmentation_mask==alphabet_dic[' ']] = 0
    return segmentation_mask
    


def adjust_overlap_box(box_output, current_index):
    """
    This function adjust the overlapping boxes.
    
    Args:
        box_output (List): List of predicted boxes.
        current_index (int): the index of current box.
    """
    
    if current_index == 0:
        return box_output
    else:
        # judge whether it contains overlap with the last output
        last_box = box_output[0, current_index-1, :]
        xmin_last, ymin_last, xmax_last, ymax_last = last_box
        
        current_box = box_output[0, current_index, :]
        xmin, ymin, xmax, ymax = current_box
        
        if xmin_last <= xmin <= xmax_last and ymin_last <= ymin <= ymax_last:
            print('adjust overlapping')
            distance_x = xmax_last - xmin
            distance_y = ymax_last - ymin
            if distance_x <= distance_y:
                # avoid overlap
                new_x_min = xmax_last + 0.025
                new_x_max = xmax - xmin + xmax_last + 0.025
                box_output[0,current_index,0] = new_x_min
                box_output[0,current_index,2] = new_x_max
            else:
                new_y_min = ymax_last + 0.025
                new_y_max = ymax - ymin + ymax_last + 0.025
                box_output[0,current_index,1] = new_y_min
                box_output[0,current_index,3] = new_y_max  
                
        elif xmin_last <= xmin <= xmax_last and ymin_last <= ymax <= ymax_last:
            print('adjust overlapping')
            new_x_min = xmax_last + 0.05
            new_x_max = xmax - xmin + xmax_last + 0.05
            box_output[0,current_index,0] = new_x_min
            box_output[0,current_index,2] = new_x_max
                    
        return box_output
    
def shrink_box(box,scale_factor=0.9):

    x1,y1,x2,y2=box
    x1_new=x1+(x2-x1)*(1-scale_factor)/2
    y1_new=y1+(y2-y1)*(1-scale_factor)/2
    x2_new=x2-(x2-x1)*(1-scale_factor)/2
    y2_new=y2-(y2-y1)*(1-scale_factor)/2

    return (x1_new,y1_new,x2_new,y2_new)

        