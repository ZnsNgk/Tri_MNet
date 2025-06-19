import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import numpy as np

class Local_Linear(nn.Module):
    def __init__(self, in_features, out_features, groups, bias=True):
        super().__init__()
        assert in_features % groups == 0
        assert out_features % groups == 0
        self.linear = nn.ModuleList()
        self.in_features = in_features//groups
        self.out_features = out_features//groups
        self.groups = groups
        for _ in range(groups):
            self.linear.append(nn.Linear(self.in_features, self.out_features, bias))
    
    def forward(self, x):
        x = torch.split(x, self.in_features, dim=-1)
        out = []
        for x_split, fc in zip(x, self.linear):
            out.append(fc(x_split).unsqueeze(-1))
        x = torch.cat(out, dim=-1)
        return x.flatten(2)

class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0., num_heads=9, use_localfc=True, local_fc_gratio=1, group_num=0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.num_heads = num_heads
        if use_localfc:
            if not group_num == 0:
                self.fc1 = Local_Linear(in_features, hidden_features, group_num)
                self.fc2 = Local_Linear(hidden_features, out_features, group_num)
            else:
                self.fc1 = Local_Linear(in_features, hidden_features, num_heads*local_fc_gratio)
                self.fc2 = Local_Linear(hidden_features, out_features, num_heads*local_fc_gratio)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        # x = x.permute(0,2,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x# x.permute(0,2,1)

class PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, kernel_size=3, in_chans=3, embed_dim=768, overlap=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=kernel_size//2)
        
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape 
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x
    
    def get_num_patches(self):
        return self.num_patches
    
# class MG_PatchEmbed(nn.Module):
#     """ (Overlapped) Image to Patch Embedding
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_heads=9, repeat_dims_to_num_heads=True):
#         '''
#         img_size (int): The size of input image
#         patch_size (int): The size image patch which you want to sperate
#         in_chans (int): The channels of input image
#         embed_dims (int): The dims of output matric
#         num_heads (int): The head num of Multi-head Attention Module
#         repeat_dims_to_num_heads(bool): Copy output matrix to num_ Heads times. If true is selected, the actual dims is dims//num_ Heads; If false is selected, it is the same as the general patch_ Embedding consistent
#         '''
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_heads = num_heads
#         self.repeat_dims_to_num_heads = repeat_dims_to_num_heads
        
#         assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
#             f"img_size {img_size} should be divided by patch_size {patch_size}."
#         assert embed_dim % num_heads == 0, "embed_dim should be divided by num_heads."
        
#         self.pool_over_imgsize = nn.ModuleList()
#         self.pool_under_imgsize = nn.ModuleList()
#         i = img_size[0]
#         self.sum_feats = 0
#         while not i == 1:
#             if i > patch_size[0]:
#                 if i == img_size[0]:
#                     self.pool_over_imgsize.append(nn.Identity())
#                 else:
#                     self.pool_over_imgsize.append(nn.AdaptiveAvgPool2d(i))
#             else:
#                 if i == img_size[0]:
#                     self.pool_under_imgsize.append(nn.Identity())
#                 else:
#                     self.pool_under_imgsize.append(nn.AdaptiveAvgPool2d(i))
#                 self.sum_feats += i**2
#             i = int(i / 2)
#         if len(self.pool_over_imgsize) > 0:
#             self.proj_over = nn.Linear(patch_size[0]*patch_size[1]*in_chans, embed_dim//num_heads if repeat_dims_to_num_heads else embed_dim)
#         if len(self.pool_under_imgsize) > 0:
#             self.proj_under = nn.Linear(self.sum_feats*in_chans, embed_dim//num_heads if repeat_dims_to_num_heads else embed_dim)
        
#         self.norm = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         shape = x.shape
#         if len(self.pool_over_imgsize) > 0:
#             feat = []
#             for p in self.pool_over_imgsize:
#                 pooled_x = p(x)
#                 if pooled_x.size(2) % self.patch_size[0] != 0:
#                     pooled_x = F.pad(pooled_x, (0, self.patch_size[0] - pooled_x.size(2) % self.patch_size[0], 0, self.patch_size[0] - pooled_x.size(2) % self.patch_size[0]))
#                 feat.append(F.unfold(pooled_x, self.patch_size, stride=self.patch_size))
#             x_over = torch.cat(feat, dim=-1).permute(0, 2, 1)
#             x_over = self.proj_over(x_over)
#         if len(self.pool_under_imgsize) > 0:
#             feat = []
#             for p in self.pool_under_imgsize:
#                 pooled_x = p(x)
#                 feat.append(pooled_x.view(shape[0], 1, -1))
#             x_under = torch.cat(feat, dim=-1)
#             x_under = self.proj_under(x_under)
#         if len(self.pool_over_imgsize) > 0 and len(self.pool_under_imgsize) > 0:
#             x = torch.cat([x_over, x_under], dim=1)
#         elif len(self.pool_over_imgsize) > 0:
#             x = x_over
#         elif len(self.pool_under_imgsize) > 0:
#             x = x_under
#         else:
#             raise ValueError('No pooling operation is performed')
#         if self.repeat_dims_to_num_heads:
#             x = x.repeat([1, 1, self.num_heads])
#         # x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x

#     def get_num_patches(self):
#         num_patches = 0
#         i = self.img_size[0]
#         while not i == 1:
#             if i > self.patch_size[0]:
#                 num_patches += (math.ceil(i/self.patch_size[0]))**2
#             else:
#                 num_patches += 1
#                 break
#             i = int(i / 2)
#         return num_patches

class MG_PatchEmbed(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_heads=9, repeat_dims_to_num_heads=True):
        '''
        img_size (int): The size of input image
        patch_size (int): The size image patch which you want to sperate
        in_chans (int): The channels of input image
        embed_dims (int): The dims of output matric
        num_heads (int): The head num of Multi-head Attention Module
        repeat_dims_to_num_heads(bool): Copy output matrix to num_ Heads times. If true is selected, the actual dims is dims//num_ Heads; If false is selected, it is the same as the general patch_ Embedding consistent
        '''
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.repeat_dims_to_num_heads = repeat_dims_to_num_heads
        
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        assert embed_dim % num_heads == 0, "embed_dim should be divided by num_heads."
        
        self.post_conv = nn.Conv2d(embed_dim//num_heads if repeat_dims_to_num_heads else embed_dim, in_chans, kernel_size=1, stride=1, padding=0)

        self.embedding_conv = nn.Conv2d(in_chans, embed_dim//num_heads if repeat_dims_to_num_heads else embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        feat = []
        flag = True # if this flag is flase, the while loop will immediate stop
        flag_2 = True   # if this flag is flase, the while loop will stop in next iteration
        while flag:
            if flag_2:
                flag = True
            else:
                flag = False
            
            # padding
            if x.shape[2] % self.patch_size[0] != 0:
                pad = self.patch_size[0] - x.shape[2] % self.patch_size[0]
                x = F.pad(x, (0, pad, 0, pad), mode='constant', value=0)
            
            # embedding
            x = self.embedding_conv(x)
            
            # flatten
            embedding = x.view(-1, x.shape[1], x.shape[2]*x.shape[3])
            
            feat.append(embedding)

            x = self.post_conv(x)

            if x.shape[2] >= self.patch_size[0]:
                flag_2 = True
            else:
                flag_2 = False

        x = torch.cat(feat, dim=-1).permute(0, 2, 1)

        if self.repeat_dims_to_num_heads:
            x = x.repeat([1, 1, self.num_heads])
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def get_num_patches(self):
        num_patches = 0
        i = self.img_size[0]
        flag = True # if this flag is flase, the while loop will immediate stop
        flag_2 = True   # if this flag is flase, the while loop will stop in next iteration
        while flag:
            if flag_2:
                flag = True
            else:
                flag = False
            
            num_patches += (math.ceil(i/self.patch_size[0]))**2
            i = int(i / self.patch_size[0])
            
            if i == 0:
                i = 1
            if i >= self.patch_size[0]:
                flag_2 = True
            else:
                flag_2 = False

            if i == 0:
                i = 1
        return num_patches

class No_Att(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
        super().__init__()
        '''移除MB Att, 其余操作与MB Att一致'''
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_scale = nn.Parameter(torch.ones(num_heads), requires_grad=True)
        self.num_attrs = num_attrs
        self.use_proj = use_proj

        # self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])
        self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])

        # self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim//num_heads, dim//num_heads)
            self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape

        x_split = torch.split(x, C // self.num_heads, dim=-1)
        q = []
        for q_split, blk_q, s in zip(x_split, self.to_q, self.q_scale):
            q_new = blk_q(q_split) * s
            q.append(q_new.reshape(B, 1, C // self.num_heads))
        q = torch.cat(q, dim=2)
        return q

class Self_Att(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
        super().__init__()
        '''将MB Att改成自注意力的模式, 其他操作不变'''
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_attrs = num_attrs
        self.use_proj = use_proj

        # self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])
        self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])
        self.to_kv = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads*2, bias=qkv_bias) for _ in range(num_heads)])

        # self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim//num_heads, dim//num_heads)
            self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape

        x_split = torch.split(x, C // self.num_heads, dim=-1)
        q_list = []
        k_list = []
        v_list = []
        for q_split, blk_q, blk_kv in zip(x_split, self.to_q, self.to_kv):
            q_new = blk_q(q_split)
            kv_new = blk_kv(q_split).reshape(B, N, 2, C // self.num_heads).permute(2, 0, 1, 3)
            q_list.append(q_new.reshape(B, 1, C // self.num_heads))
            k_list.append(kv_new[0].reshape(B, 1, C // self.num_heads))
            v_list.append(kv_new[1].reshape(B, 1, C // self.num_heads))
        
        out = []
        for q, k, v in zip(q_list, k_list, v_list):
            attn = (q.transpose(-2, -1) @ k) * self.scale
            attn = attn.softmax(dim=-1)
            out.append((attn @ v.transpose(-2, -1)).transpose(-2, -1))
        
        x = torch.cat(out, dim=-1)
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        if self.use_proj:
            x = self.proj(x)
        
        return x

class Memory_Bank_Att(nn.Module):
    # 作为MHSA使用, 可以用于多个block, 不改变tokens数量
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_proj=True, local_fc_gratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_proj = use_proj

        self.to_k = Local_Linear(dim, dim, groups=num_heads*local_fc_gratio, bias=qkv_bias)
        self.to_v = Local_Linear(dim, dim, groups=num_heads*local_fc_gratio, bias=qkv_bias)

        if self.use_proj:
            self.proj = Local_Linear(dim, dim, groups=num_heads)

        # 使用正态分布初始化 memory_bank，提升区分性
        self.memory_bank = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02, requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape  # B: 批量大小, N: 序列长度, C: 特征维度

        # 分割输入到每个注意力头
        k = self.to_k(x).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        v = self.to_v(x).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)

        # 初始化查询向量
        k = k.transpose(-2, -1)  # (B, num_heads, head_dim, N)

        # att
        attn = torch.matmul(self.memory_bank.unsqueeze(-2).repeat(1, N, 1), k) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (B, num_heads, head_dim)

        # out = out.view(B, self.num_heads, -1)  # 拼接注意力头的输出

        x = out.contiguous().view(B, N, -1)  # 拼接所有 head 的输出

        if self.use_proj:
            x = self.proj(x)  # 线性投影

        return x
    
class Memory_Bank_Emb_Att(nn.Module):
    # 作为FFN使用, 可以用于多个block, 不改变tokens数量
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=1, use_proj=True, act_layer=nn.GELU):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or num_patches ** -0.5
        self.use_proj = use_proj

        self.to_k = nn.Linear(num_patches, num_patches)
        self.to_v = nn.Linear(num_patches, num_patches)
        self.act = act_layer()

        if self.use_proj:
            self.proj = nn.Linear(num_patches, num_patches)

        # 使用正态分布初始化 memory_bank，提升区分性
        self.memory_bank = nn.Parameter(torch.randn(num_heads, head_dim) * 0.02, requires_grad=True)

    def forward(self, x):
        B, N, C = x.shape  # B: 批量大小, N: 序列长度, C: 特征维度
        x = x.transpose(-2, -1)  # (B, C, N)

        # 分割输入到每个注意力头
        k = self.to_k(x).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        v = self.to_v(x).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)

        # 初始化查询向量
        k = k.transpose(-2, -1)  # (B, num_heads, N, head_dim)

        attn = torch.matmul(self.memory_bank.unsqueeze(-2).repeat(1, N, 1), k) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).squeeze(-1)  # (B, num_heads, head_dim)
        out = self.act(out)
        out = out.permute(0, 2, 1, 3)   # (B, N, num_heads, head_dim)

        # out = out.view(B, self.num_heads, -1)  # 拼接注意力头的输出

        x = out.contiguous().view(B, N, -1)  # 拼接所有 head 的输出

        if self.use_proj:
            x = self.proj(x.transpose(-2, -1)).transpose(-2, -1)  # 线性投影

        return x

# class Memory_Bank_Emb_Att(nn.Module):
#     def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.q_scale = nn.Parameter(torch.ones(num_heads) * 0.1, requires_grad=True)  # 初始化为较小值以提升稳定性
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_attrs = num_attrs
#         self.use_proj = use_proj

#         self.to_q = nn.ModuleList([nn.Linear(head_dim, head_dim, bias=qkv_bias) for _ in range(num_heads)])

#         if self.use_proj:
#             self.proj = Local_Linear(dim, dim, groups=num_heads)

#         # 使用正态分布初始化 memory_bank，提升区分性
#         self.memory_bank = nn.Parameter(torch.randn(num_heads, num_attrs, head_dim) * 0.02, requires_grad=True)

#     def forward(self, x):
#         B, N, C = x.shape  # B: 批量大小, N: 序列长度, C: 特征维度

#         # 分割输入到每个注意力头
#         x_split = torch.split(x, C // self.num_heads, dim=-1)
#         q = []
#         for x_s, blk_q, s_q in zip(x_split, self.to_q, self.q_scale):
#             q_new = blk_q(x_s) * s_q  # 使用可学习的缩放因子
#             q.append(q_new.reshape(B, N, 1, C // self.num_heads))
            
#         q = torch.cat(q, dim=2)  # (B, N, num_heads, head_dim)

#         # 初始化查询向量
#         q = q.permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)

#         # 初始化输出
#         out = torch.zeros_like(q).repeat(1, 1, self.num_attrs, 1)
#         q_t = q.transpose(-2, -1)  # (B, num_heads, head_dim, N)

#         for i in range(self.num_attrs):
#             # emb att
#             attn = torch.matmul(q_t, self.memory_bank[:, i, :].unsqueeze(-2)) * self.scale  # (B, num_heads, N, 1)
#             attn = attn.softmax(dim=-1)
#             o = torch.matmul(attn, q_t).squeeze(-1)  # (B, num_heads, head_dim)
#             out[:, :, i, :] = o # (B, num_heads, num_attrs, head_dim)

#         # out = out.view(B, self.num_heads, -1)  # 拼接注意力头的输出

#         x = out.contiguous()#.view(B, self.num_attrs, -1)  # 拼接所有 head 的输出

#         if self.use_proj:
#             x = self.proj(x)  # 线性投影

#         return x

# class Memory_Bank_Emb_Att(nn.Module):
#     def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.q_scale = nn.Parameter(torch.ones(num_heads), requires_grad=True)  # Moderate initialization for stability and performance
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_attrs = num_attrs
#         self.use_proj = use_proj

#         self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])  # 键向量的线性变换

#         if self.use_proj:
#             self.proj = nn.Linear(dim//num_heads, dim//num_heads)

#         self.memory_bank = nn.Parameter(torch.ones(num_heads, num_attrs, head_dim), requires_grad=True)  # Moderate initialization to improve stability

#     def forward(self, x):
#         B, N, C = x.shape  # B: 批量大小, N: 序列长度, C: 特征维度

#         x_split = torch.split(x, C // self.num_heads, dim=-1)
#         q = []
#         for x_s, blk_q, s_q in zip(x_split, self.to_q, self.q_scale):
#             q_new = blk_q(x_s) * s_q
#             q.append(q_new.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3))
            
#         q = torch.cat(q, dim=1)

#         # 初始化输出
#         out = torch.zeros_like(q).repeat(1, 1, self.num_attrs, 1)
#         q = q.transpose(-2, -1)

#         for i in range(self.num_attrs):
#             # emb att
#             attn = torch.matmul(q, self.memory_bank[:, i, :].unsqueeze(-2)) * self.scale  # (B, num_heads, head_dim, head_dim)
#             attn = attn.softmax(dim=-1)
#             o = torch.matmul(attn, q).squeeze(-1)
#             out[:, :, i, :] = o
        
#         x = out.contiguous()
#         if self.use_proj:
#             x = self.proj(x)
#         return x
    
# class Memory_Bank_Emb_Att(nn.Module): # use mb to Q, and generatr kv
#     # faster
#     def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.k_scale = nn.Parameter(torch.ones(num_heads), requires_grad=True)  # Moderate initialization for stability and performance
#         self.v_scale = nn.Parameter(torch.ones(num_heads), requires_grad=True)  # Moderate initialization for stability and performance
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_attrs = num_attrs
#         self.use_proj = use_proj

#         self.to_k = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])  # 键向量的线性变换
#         self.to_v = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])  # 值向量的线性变换

#         if self.use_proj:
#             self.proj = nn.Linear(dim//num_heads, dim//num_heads)

#         self.memory_bank = nn.Parameter(torch.ones(num_heads, num_attrs, head_dim), requires_grad=True)  # Moderate initialization to improve stability

#     def forward(self, x):
#         B, N, C = x.shape  # B: 批量大小, N: 序列长度, C: 特征维度

#         x_split = torch.split(x, C // self.num_heads, dim=-1)
#         k = []
#         v = []
#         for x_s, blk_k, s_k, blk_v, s_v in zip(x_split, self.to_k, self.k_scale, self.to_v, self.v_scale):
#             k_new = blk_k(x_s) * s_k
#             k.append(k_new.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3))
#             v_new = blk_v(x_s) * s_v
#             v.append(v_new.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3))
#         k = torch.cat(k, dim=1)
#         v = torch.cat(v, dim=1)

#         # 初始化输出
#         out = torch.zeros_like(k).repeat(1, 1, self.num_attrs, 1)
#         v = v.transpose(-2, -1)

#         for i in range(self.num_attrs):
#             # emb att
#             attn = torch.matmul(self.memory_bank[:, i, :].unsqueeze(-2).transpose(-2, -1), k) * self.scale  # (B, num_heads, head_dim, head_dim)
#             attn = attn.softmax(dim=-1)
#             o = torch.matmul(attn, v).squeeze(-1)
#             out[:, :, i, :] = o
        
#         x = out.contiguous()
#         if self.use_proj:
#             x = self.proj(x)
#         return x
    
# class Memory_Bank_Emb_Att(nn.Module):
#     def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_attrs=1, use_proj=True):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.q_scale = nn.Parameter(torch.ones(num_heads), requires_grad=True)
#         self.scale = qk_scale or head_dim ** -0.5
#         self.num_attrs = num_attrs
#         self.use_proj = use_proj

#         # self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])
#         self.to_q = nn.ModuleList([nn.Linear(dim//num_heads, dim//num_heads, bias=qkv_bias) for _ in range(num_heads)])

#         # self.attn_drop = nn.Dropout(attn_drop)
#         if self.use_proj:
#             self.proj = nn.Linear(dim//num_heads, dim//num_heads)
#             self.proj_drop = nn.Dropout(proj_drop)
        
#         # self.norm = nn.LayerNorm(dim)

#         # self.memory_bank = nn.Parameter(torch.ones(2, num_heads, num_attrs, dim//num_heads), requires_grad=True)
#         self.memory_bank = nn.Parameter(torch.ones(num_heads, num_attrs, dim//num_heads), requires_grad=True)
#         # self.register_parameter("Memory_Bank", self.memory_bank)
        
#     def forward(self, x):
#         B, N, C = x.shape

#         x_split = torch.split(x, C // self.num_heads, dim=-1)
#         q = []
#         for q_split, blk_q, s in zip(x_split, self.to_q, self.q_scale):
#             q_new = blk_q(q_split) * s
#             q.append(q_new.reshape(B, -1, 1, C // self.num_heads).permute(0, 2, 1, 3))
#         outs = []
    
#         q_mem = torch.split(self.memory_bank, 1, dim=0)
        
#         for q_split, q_m in zip(q, q_mem):
#             out = []
#             q_numattrs = torch.split(q_m, 1, dim=1)
#             for q_n in q_numattrs:
#                 # 传统self-attention
#                 a = (q_split.transpose(-2, -1) @ q_n) * self.scale
#                 a = a.softmax(dim=-1)
#                 o = (a @ q_split.transpose(-2, -1)).transpose(-2, -1).squeeze(3)
#                 out.append(o)
#             out = torch.cat(out, dim=2)
#             outs.append(out)
#         out = torch.cat(outs, dim=1)

#         # q = torch.cat(q, dim=2)
#         # k, v = self.memory_bank[0], self.memory_bank[1]
#         # k = torch.split(k, 1, dim=0)
#         # v = torch.split(v, 1, dim=0)
#         # for q_split, k_numheads, v_numheads in zip(q, k, v):
#         #     out = []
#         #     k_numattrs = torch.split(k_numheads, 1, dim=1)
#         #     v_numattrs = torch.split(v_numheads, 1, dim=1)
#         #     for k_split, v_split in zip(k_numattrs, v_numattrs):
#         #         a = (q_split.transpose(-2, -1) @ k_split) * self.scale
#         #         a = a.softmax(dim=-1)
#         #         o = (a @ v_split.transpose(-2, -1)).transpose(-2, -1).squeeze(3)
#         #         out.append(o)
#         #     out = torch.cat(out, dim=2)
#         #     outs.append(out)
#         # out = torch.cat(outs, dim=1)
        
#         x = out.contiguous()
#         if self.use_proj:
#             x = self.proj(x)
#         return x

# class Generate_Attr_Relation_Graph(nn.Module):
#     def __init__(self, num_class, num_attr, num_density=6):
#         super().__init__()
#         self.num_attr = num_attr
#         self.leading_matrix = nn.Parameter(torch.eye(num_class) * np.sqrt(self.num_attr) * np.sqrt(num_class), requires_grad=True)
#         self.num_density = num_density
#         # 使用可学习的参数作为概率密度权重, 仅用于两个得分的属性
#         self.normal_distribution_probability_density = nn.Parameter(torch.Tensor([0.3989, 0.2420, 0.0540, 0.0044, 0., 0.]), requires_grad=True)

#     def forward(self, res):
#         dev = res.device
#         # argmax无法求导, 采用其他方法获得标签
#         # lab = torch.argmax(res, dim=-1)
        
#         logit = torch.softmax(res, dim=-1)
#         indices = torch.arange(self.num_attr, device=dev, dtype=torch.float)
#         lab = logit @ indices
        
#         leading_matrix = F.softmax(self.leading_matrix, dim=0)
#         rel_lab = lab @ leading_matrix

#         batch_weights = []
#         # 将两个得分转换为每个分数的概率密度
#         for B in rel_lab:
#             # 计算相对标签与绝对标签之间的距离
#             dist = torch.abs(torch.arange(self.num_attr, device=dev).unsqueeze(0).float() - B.unsqueeze(-1).float())
#             # 计算概率密度
#             weight = self.normal_distribution_probability_density * torch.exp(-dist)
#             # weight /= torch.sum(weight, dim=-1, keepdim=True)
#             batch_weights.append(weight.unsqueeze(0))

#         batch_weights = torch.cat(batch_weights, dim=0)
#         out = res * batch_weights
#         return out

class Generate_Attr_Relation_Graph(nn.Module):
    def __init__(self, num_heads, num_attr, output_classes, residual_graph=False):
        super().__init__()
        self.num_node = num_heads*num_attr
        self.num_attr = num_attr
        self.output_classes = output_classes
        self.residual_graph = residual_graph
        # self.rel = nn.Linear(num_heads*num_attr, output_classes, bias=False)
        self.leading_matrix = nn.Parameter(torch.ones([self.num_node + output_classes]*2) * np.sqrt(self.num_node), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor([output_classes]), requires_grad=True)
        # 使用可学习的参数作为概率密度权重, 仅用于得分的属性
        # self.normal_distribution_probability_density = nn.Parameter(torch.randn([output_classes]), requires_grad=True)

    def forward(self, res):
        dev = res.device
        # 添加需要预测的类别
        # last_rel = self.rel(res.view(res.shape[0], -1))
        last_rel = torch.ones([res.shape[0], self.output_classes], device=dev) * self.scale
        
        # argmax无法求导, 采用其他方法获得标签
        # lab = torch.argmax(res, dim=-1)
        # leading_matrix = (self.leading_matrix.T + self.leading_matrix)/2.
        leading_matrix = self.leading_matrix
        
        # # 获取得分, 并计算软标签
        # score = res[:, -1, :].unsqueeze(1)
        # logit = torch.softmax(score, dim=-1)
        # indices = torch.arange(self.num_attr, device=dev, dtype=torch.float)
        # lab = logit @ indices
        # lab = lab / self.num_attr  #缩放到0-1范围内
        # # 将剩余标签拼接
        # score = torch.softmax(res[:, :-1, :], dim=-1).view(-1, self.num_node)
        # lab = torch.cat([score, lab], dim=-1)
        
        leading_matrix = F.softmax(leading_matrix, dim=-1)
        leading_matrix = torch.clamp(leading_matrix, min=1e-6, max=1.0)

        rel_lab = torch.cat([res.view(res.shape[0], -1), last_rel], dim=-1) @ leading_matrix
        rel_lab_prev = rel_lab[:, :-self.output_classes].view(res.shape)
        rel_lab_last = rel_lab[:, -self.output_classes:]


        # batch_weights = []
        # # 将最后一个得分转换为每个分数的概率密度
        # for B in rel_lab:
        #     # 计算相对标签与绝对标签之间的距离
        #     dist = torch.abs(torch.arange(self.num_attr, device=dev).unsqueeze(0).float() - (B[-1].unsqueeze(-1).float()*float(self.num_attr)))
        #     # 计算概率密度
        #     weight = self.normal_distribution_probability_density * torch.exp(-dist)
        #     # weight /= torch.sum(weight, dim=-1, keepdim=True)
        #     other_weights = B[:-1].view(-1, self.num_attr)
        #     weight = torch.cat([other_weights, weight], dim=0)
        #     batch_weights.append(weight.unsqueeze(0))

        # batch_weights = torch.cat(batch_weights, dim=0)
        prev_out = res * rel_lab_prev
        last_out = last_rel * rel_lab_last
        if self.residual_graph:
            prev_out = prev_out + res
        return last_out, prev_out

class Emb_Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_proj=True, use_localfc=False, local_fc_gratio=1):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_proj = use_proj
        if use_localfc:
            self.q = nn.Sequential(Local_Linear(dim, dim, bias=qkv_bias, groups=num_heads*local_fc_gratio))
            self.kv = nn.Sequential(Local_Linear(dim, dim * 2, bias=qkv_bias, groups=num_heads*local_fc_gratio))
        else:
            self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
            self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        if self.use_proj:
            x = self.proj(x)

        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_proj=True, use_localfc=False, local_fc_gratio=1, group_num=0):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_proj = use_proj

        if use_localfc:
            if not group_num == 0:
                self.q = nn.Sequential(Local_Linear(dim, dim, bias=qkv_bias, groups=group_num))
                self.kv = nn.Sequential(Local_Linear(dim, dim * 2, bias=qkv_bias, groups=group_num))
            else:
                self.q = nn.Sequential(Local_Linear(dim, dim, bias=qkv_bias, groups=num_heads*local_fc_gratio))
                self.kv = nn.Sequential(Local_Linear(dim, dim * 2, bias=qkv_bias, groups=num_heads*local_fc_gratio))
        else:
            self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
            self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        
        self.attn_drop = nn.Dropout(attn_drop)
        if self.use_proj:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)   
        x = x.transpose(1,2).contiguous().reshape(B, N, C)
        if self.use_proj:
            x = self.proj(x)

        return x
