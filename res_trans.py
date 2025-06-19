# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:58:52 2024

@author: BHN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Type, Union
from einops import rearrange, reduce, repeat
from timm.models.registry import register_model

class F2T(nn.Module):
    def __init__(self, p, in_planes, embedding_dimension):
        super().__init__()
        self.p = p
        self.embedding_dimension = embedding_dimension
        self.conv1 = nn.Conv2d(in_planes, embedding_dimension, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(embedding_dimension)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((p, p))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dimension))
        # self.position = nn.Parameter(torch.arange(self.p*self.p+1, device='cuda'))
        # position = torch.arange(self.p*self.p+1, dtype=torch.float32)
        # position = torch.reshape(position,(1,1,-1))
        # position = position.expand(-1,self.embedding_dimension,-1)
        # self.position = nn.Parameter(position)
        self.position = nn.Parameter(torch.randn(1, self.p*self.p+1  ,embedding_dimension) )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.reshape(x,((x.shape[0],self.embedding_dimension,self.p*self.p)))
        x = torch.transpose(x, 1, 2)
        cls_token = repeat(self.cls_token, "() n d -> b n d",b=x.shape[0]) 
        # self.cls_token = self.cls_token.expand(x.shape[0],-1,-1)
        x = torch.cat((cls_token,x), dim=1)
        
        x = self.position + x
        
        return x
    
class MHSA(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c):
        super().__init__()
        num_heads = c//2
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c*2, bias=False)
        self.fc2 = nn.Linear(c*2, c, bias=False)
        self.Norm = nn.LayerNorm(c)

    def forward(self, x):
        x = self.Norm(x) 
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.Norm(x)
        x = self.fc2(self.fc1(x)) + x
        return x

class T2F(nn.Module):
    def __init__(self, in_planes, out_planes, downsample):
        super().__init__()
        self.in_planes = in_planes
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = x[:,1:,:]
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x,((x.shape[0],self.in_planes,int(x.shape[2]**0.5),int(x.shape[2]**0.5))))
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        if not self.downsample:
            x = self.up(x)
        return x

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample_flag = False,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride = 1, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride = stride, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = nn.Conv2d(inplanes, planes, kernel_size=1, stride = stride, padding=0)
        self.downsample_flag = downsample_flag
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample_flag:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Res_trans(nn.Module):
    def __init__(self, num_classes, in_chans=1, **kwargs):
        super().__init__()
        self.layer_1 = nn.Sequential(BasicBlock(inplanes=32,planes=32),
                                     BasicBlock(inplanes=32,planes=32))
        self.layer_2 = nn.Sequential(F2T(p=8,in_planes=32, embedding_dimension=32),
                                     MHSA(32),
                                     MHSA(32),
                                     MHSA(32),
                                     MHSA(32),
                                     T2F(32,32,downsample=False))
        self.layer_3 = nn.Sequential(BasicBlock(inplanes=32,planes=64,stride=2,downsample_flag=True),
                                     BasicBlock(inplanes=64,planes=64))
        self.layer_4 = nn.Sequential(F2T(p=4,in_planes=64, embedding_dimension=64),
                                     MHSA(64),
                                     MHSA(64),
                                     MHSA(64),
                                     MHSA(64),
                                     T2F(64,64,downsample=False))
        
        self.conv1 = nn.Conv2d(in_chans, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self.identity = nn.Identity()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        
        attention = self.conv2(x)
        attention = self.sigmoid(attention)
        x = x*attention
        x = self.identity(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

@register_model    
def res_trans(pretrain=False, **kwargs):
    return Res_trans(**kwargs)

if __name__ == "__main__":
    model = res_trans(False, num_classes=6).cuda()
    img = torch.randn([1, 1, 64, 64]).cuda()
    out = model(img)
    print(out.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=(img,))
        flops, params = clever_format([flops, params], "%.6f")
    print("Flops: ", flops)
    print("num_paras: ", params)