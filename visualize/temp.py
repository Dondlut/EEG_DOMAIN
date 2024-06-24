import torch
import torch.nn as nn
from model import DANNModel
from utils import GradCAM, show_cam_on_image, center_crop_img
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import h5py
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体


def plotCam(h5_path):
    model_pre = DANNModel(nb_classes=4, dropOut=None)
    model_pre.load_state_dict(torch.load('./model/model.pth', map_location=torch.device('cpu')), False)
    target_layers = [model_pre.feature.dense_block_final['number 3 ConvBlock'].convSST]
    fileName = h5_path.split("/")[-1]

    with h5py.File(h5_path, 'r') as hf:
        input_tensor = hf['clip'][()]
        input_tensor = torch.unsqueeze(torch.tensor(input_tensor), dim=0).float()
        input_tensor = input_tensor.permute(0, 3, 1, 2)
        input_tensor = torch.unsqueeze(input_tensor, dim=0)
    target_category = int(h5_path.split("_")[-1].split(".")[0])

    cam = GradCAM(model=model_pre, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(np.array(input_tensor),
                                      grayscale_cam,
                                      use_rgb=True)
    # visualization = np.transpose(visualization,[1,0,2])
    plt.xlabel('频率(Hz)')
    plt.ylabel('通道(编号)')
    x_ticks = [0,int(20 / 0.775),int(40 / 0.775),int(60 / 0.775),int(80 / 0.775),129]
    x_labels = [0,20,40,60,80,100]
    plt.xticks(x_ticks,x_labels)
    plt.imshow(visualization)
    plt.imshow(visualization)
    plt.savefig('./fig/{}/{}.jpg'.format(target_category, fileName), bbox_inches='tight', pad_inches=0.1)
    plt.show()


if __name__ == '__main__':

    h5_path = r'./h5_anno/0/00008476_s006_t008.edf_0_20_0.h5'

    # h5_path = r'./h5_anno/1/00009231_s002_t000.edf_0_40_1.h5'

    # h5_path = r'./h5_anno/2/00001413_s002_t001.edf_0_1_2.h5'

    # h5_path = r'./h5_anno/3/00009044_s001_t000.edf_0_8_3.h5'


    plotCam(h5_path)
