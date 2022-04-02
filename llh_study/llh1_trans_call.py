#%%
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms#注意查看Normalize和Resize的说明，归一化和尺度变化
import cv2
#%%
#__call__ 赋予class函数的能力，即可以直接当函数使用该对象
img_path = 'shushu.png'
img_1 = cv2.imread(img_path)
tf = transforms.ToTensor()
img1_tensor = tf(img_1)
tu_board = SummaryWriter('shushu')
tu_board.add_image('shushu',img1_tensor)
# %%
#Normalize
guiyi = transforms.Normalize([0.5,0.5,0.5],[3,2,1])#2*std-1
img1_guiyi = guiyi(img1_tensor)
tu_board.add_image('shushu2',img1_guiyi)
#%%
#compose
#%%
tu_board.close()