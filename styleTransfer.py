import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from synthesizedImage import synthesizedImage
from tqdm import tqdm
from Loss import *
from torch.optim import Adam


rgb_mean = torch.Tensor([0.485, 0.456, 0.406])
rgb_std = torch.Tensor([0.229, 0.224, 0.225])


class styleTransfer():
    def __init__(self, style_layer, content_layer, device):
        super().__init__()

        #指定需要在CNN中提取特征图的层
        self.style_layer = style_layer
        self.content_layer = content_layer
        #预训练的CNN用于提取特征
        pretrained_net = torchvision.models.vgg19(pretrained=True) 
        self.net = nn.Sequential(
            *[
                pretrained_net.features[i] for i in range(max(style_layer + content_layer) + 1)
            ]
        ) # * 可以把list,tuple分成单独的元素传给函数
        self.net.requires_grad_ = False
        self.net.to(device)

        self.device = device

    #进入模型前预处理图片
    def img_preprocess(self, img, img_shape):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_shape),
            torchvision.transforms.ToTensor(),#转换为tensor时会除以255
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)
        ])
        return transform(img).unsqueeze(0)

    #img_preprocess的逆操作
    def img_postprocess(self, img):
        img = img[0].to(rgb_std.device)
        img = torch.clamp(
            img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1 #需要把channel放在第三维才能运算
        )
        return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1)) #把channel放在第一维再传给ToPILImage()

    def extract_feature(self, img): #CNN提取特征，取特定层的特征图
        style_feature = []
        content_feature = []
        for i in range(len(self.net)):
            img = self.net[i](img)
            if i in self.style_layer:
                style_feature.append(img)
            elif i in self.content_layer:
                content_feature.append(img)
        return style_feature, content_feature

    #初始化，传入待处理的图片和目标风格的图片
    def init(self, content_pic, style_pic, img_shape):
        self.style_x = self.img_preprocess(style_pic, img_shape).to(self.device)
        self.style_y, _ = self.extract_feature(self.style_x)#提前抽取特征，避免重复计算

        self.content_x = self.img_preprocess(content_pic, img_shape).to(self.device)
        _, self.content_y = self.extract_feature(self.content_x)#提前抽取特征，避免重复计算

        self.synthesizedImage = synthesizedImage(self.content_x.shape, self.device, self.content_x)#将图片作为模型权重

    
    def train(self, epochs, lr, plot_loss=False):
        content_ls, style_ls, total_ls = [], [], []
        style_gram_Y = [gram(y) for y in self.style_y]#提前处理，避免重复计算
        optimizer = Adam(self.synthesizedImage.parameters(), lr)
        for i in tqdm(range(epochs)):
            optimizer.zero_grad()
            #self.synthesizedImage()表示正在进行风格迁移的图片
            style_hat, content_hat = self.extract_feature(self.synthesizedImage())#抽取特征
            content_l, style_l, tv_l, total_l = compute_loss(#将当前的特征与原图片、风格图片的特征分别计算loss
                self.synthesizedImage(), content_hat, style_hat, self.content_y, style_gram_Y
            )
            total_l.backward()
            optimizer.step()
            #print("epoch %d-----total_loss=%.3f"%(i, total_l.item()))
            content_ls.append(sum(content_l).item())
            style_ls.append(sum(style_l).item())
            total_ls.append(total_l.item())
        
        if plot_loss:
            plt.plot(content_ls, label="content_loss")
            plt.plot(style_ls, label="style_ls")
            plt.plot(total_ls, label="total_ls")
            plt.legend()
            plt.savefig("./train_loss.png", dpi=300)

    def getResult(self):
        return self.img_postprocess(self.synthesizedImage())





    

    