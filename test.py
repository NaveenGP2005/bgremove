import requests
import torch
import numpy as np
from torch.autograd import Variable
from PIL import Image

from u2net import U2NET
import datetime

# weights can be downoloaded here
# https://github.com/xuebinqin/U-2-Net 
model_file = r"C:\cv\u2net_app\weights\u2net.pth"
filename = r"C:\cv\u2net_app\static\results\Image1.jpg"
outfile = r"C:\DriveR\Rubick\bg_work\u2net\peko\img\output\Queenly_A-DIA-ANA-01\output_nin.png"
mask_file = r"C:\DriveR\Rubick\bg_work\u2net\peko\img\output\Queenly_A-DIA-ANA-01\mask_queen\mask_gph.png"
input_size = 320

# load image from file or url
def load_image(file_or_url_or_path):
    if isinstance(file_or_url_or_path, str) and file_or_url_or_path.startswith("http"):
        file_or_url_or_path = requests.get(file_or_url_or_path, stream=True).raw
    return Image.open(file_or_url_or_path)


# convert image for use in model pillow -> torch
def convert_image(pillow_image):

    # resize & convert to rgb
    image = pillow_image.resize((input_size, input_size))
    image = image.convert('RGB')
    # pillow -> numpy
    image = np.array(image)
    # convert to LAB 
    tmpImg = np.zeros((image.shape[0], image.shape[1],3))
    # normalize
    image = image/np.max(image) 
    tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
    tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
    tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225
    # reshape (320,320,3) -> (3,320,320)
    tmpImg = tmpImg.transpose((2, 0, 1))
    # reshape (3,320,320)  -> (1, 3, 320, 320)
    tmpImg = tmpImg[np.newaxis,:,:,:]
    # numpy -> torch
    image = torch.from_numpy(tmpImg)
    image = image.type(torch.FloatTensor)
    image = Variable(image)
    
    return image


# Normalize tensor (torch version)
def normalize(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


# save results
def save_output(image, mask, out_path=None):
    # Normalize mask to [0, 1]
    mask = normalize(mask)
    
    # Squeeze and scale to [0, 255]
    mask = mask.squeeze()
    mask = mask.cpu().data.numpy() * 255
    
    # Convert to grayscale PIL image
    mask = Image.fromarray(mask).convert("L")
    
    # Resize mask to match original image
    mask = mask.resize(image.size, resample=Image.Resampling.BILINEAR)
    
    # Convert image to RGBA and apply alpha mask
    image = image.convert('RGBA')
    image.putalpha(mask)

    # Save image if output path is provided
    if out_path:
        image.save(out_path)

    return image


def main():

    # init model
    net = U2NET(3,1)
    net.load_state_dict(torch.load(model_file, map_location=torch.device('cuda')))
    net.eval()
    
    # load image
    pillow_image = load_image(filename)
    torch_image = convert_image(pillow_image) 
    # Test, result shape must be (1,3,320,320)
    # print(image.shape)

    # feed to model
    start = datetime.datetime.now()
    with torch.no_grad():
        d1,d2,d3,d4,d5,d6,d7 = net(torch_image)
    end = datetime.datetime.now()
    differ = start - end
    print(f'Time taken: {differ.microseconds} microseconds')
    # recieve d1 mask
    mask = d1[:,0,:,:]    
    # save result
    save_output(pillow_image, mask, outfile)

    # cleanup
    del d1,d2,d3,d4,d5,d6,d7

if __name__ == '__main__':
    main()
