import os
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import torchvision
from PIL import Image


class MnistDataset(Dataset):
    """ created datatset class rather than using torchvision to allow replacement with other image dataset"""

    def __init__(self,split,im_path,im_ext='png'):


        self.split=split
        self.im_ext=im_ext
        self.images,self.labels=self.load_images(im_path)

    def load_images(self,im_path):
        """Gets all images from the path specified and stacks them all up"""

        assert os.path.exists(im_path),"images path {} does not exists".format(im_path)
        ims=[]
        labels=[]
        for d_name in tqdm(os.listdir(im_path)):
            for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))

        
        print('Found {} images for split {}'.format(len(ims),self.split))

        return ims,labels
    

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,index):
        im=Image.open(self.images[index])
        im_tensor=torchvision.transforms.ToTensor()(im)

        # Convert input to -1 to 1 range
        im_tensor=(2*im_tensor)-1
        return im_tensor



