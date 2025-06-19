from os import sep
from pickle import TRUE
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math
import numpy as np
try:
    from . import modules
except:
    import modules

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., num_patches=1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_attrs=1, use_IRB=False, att_type="mbatt", use_localfc=True, local_fc_gratio=1, use_proj=False, mlp_type="mlp", group_num=0):
        super().__init__()
        self.num_heads = num_heads
        self.use_IRB = use_IRB
        self.att_type = att_type
        self.num_attrs = num_attrs
        self.norm1 = norm_layer(dim)
        if self.att_type == "mbatt":
            self.attn = modules.Memory_Bank_Att(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj, local_fc_gratio=local_fc_gratio)
        elif self.att_type == "att":
            self.attn = modules.Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj, use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, group_num=group_num)
        elif self.att_type == "embatt":
            self.attn = modules.Emb_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj, use_localfc=use_localfc, local_fc_gratio=local_fc_gratio)
        elif self.att_type == "noatt":
            self.attn = modules.No_Att(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj)
        elif self.att_type == "selfatt":
            self.attn = modules.Self_Att(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj)
        else:
            raise NotImplementedError
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.use_IRB:
            self.norm2 = norm_layer(dim)
            if mlp_type == "mbatt":
                self.mlp = modules.Memory_Bank_Emb_Att(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, act_layer=nn.Hardswish,
                    attn_drop=attn_drop, proj_drop=drop, use_proj=use_proj, num_patches=num_patches)
            elif mlp_type == "mlp":
                self.mlp = modules.IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), num_heads=num_heads, act_layer=nn.Hardswish, drop=drop, ksize=3, use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, group_num=group_num)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.use_IRB:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MG_Memorybank_ATT_ViT(nn.Module):
    def __init__(self, img_size=64, patch_size=64, in_chans=1, num_classes=5, embed_dims=4320, num_attrs=6,
                 num_heads=7, mlp_ratios=1., qkv_bias=True, qk_scale=None, drop_rate=0., local_fc_gratio=1, 
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6), group_num=0,
                 depths=1, pemb_type="mgpemb", use_IRB=False, att_type="mbatt", use_localfc=True, use_proj=False, repeat_emb=True,
                 use_graph=False, residual_graph=False, use_mask=False, use_DS=False, eda_type="", **kwargs):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_attrs = num_attrs
        self.att_type = att_type
        self.repeat_emb = repeat_emb
        self.use_graph = use_graph
        self.use_mask = use_mask
        self.use_DS = use_DS
        if self.use_mask:
            self.mask = torch.Tensor([[1., 1., 0., 1., 1., 0.],
                                    [1., 1., 1., 1., 1., 1.],
                                    [1., 1., 1., 1., 1., 0.],
                                    [1., 1., 1., 1., 1., 0.],
                                    [1., 1., 1., 1., 1., 0.],
                                    [1., 1., 1., 1., 1., 0.],
                                    [1., 1., 1., 1., 1., 0.]])

        if pemb_type == "mgpemb":
            self.patch_embed = modules.MG_PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims, repeat_dims_to_num_heads=repeat_emb, num_heads=num_heads)
        elif pemb_type == "pemb":
            self.patch_embed = modules.PatchEmbed(img_size=img_size, patch_size=patch_size, kernel_size=3, in_chans=in_chans,
                                       embed_dim=embed_dims, overlap=False)
        else: 
            raise NotImplementedError
        num_patches = self.patch_embed.get_num_patches()
        
        self.pos_embeddings = nn.Parameter(torch.zeros(1, num_patches, self.embed_dims))
        
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        if eda_type == "no":    # no EDA
            self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, num_patches=num_patches,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_attrs=num_attrs, use_IRB=use_IRB,
            use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, use_proj=use_proj, att_type=att_type, mlp_type="mlp", group_num=group_num)
            for i in range(depths)])
        elif eda_type == "all":    # all EDA
            self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, num_patches=num_patches,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_attrs=num_attrs, use_IRB=use_IRB,
            use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, use_proj=use_proj, att_type=att_type, mlp_type="mbatt", group_num=group_num)
            for i in range(depths)])
        elif "num" in eda_type:    # num EDA
            self.block = nn.ModuleList()
            num = int(eda_type.split("_")[1])   # num_x (x<12)
            assert not num > depths
            replace_flag = depths - num
            for i in range(depths):
                if i < replace_flag:
                    self.block.append(Block(
                    dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, num_patches=num_patches,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_attrs=num_attrs, use_IRB=use_IRB,
                    use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, use_proj=use_proj, att_type=att_type, mlp_type="mlp", group_num=group_num))
                else:
                    self.block.append(Block(
                    dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, qk_scale=qk_scale, num_patches=num_patches,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_attrs=num_attrs, use_IRB=use_IRB,
                    use_localfc=use_localfc, local_fc_gratio=local_fc_gratio, use_proj=use_proj, att_type=att_type, mlp_type="mbatt", group_num=group_num))
        else:
            raise NotImplementedError
        
        # classification head
        # self.head = nn.Linear(embed_dims[2], num_classes) if num_classes > 0 else nn.Identity()
        # Multi classification heads
        if not (att_type == "mbatt" or att_type == "mattratt"):
            if repeat_emb:
                self.heads = nn.ModuleList()
                for _ in range(num_heads):
                    self.heads.append(nn.Linear(embed_dims//num_heads, num_attrs))
                self.rel = nn.Linear(num_attrs*num_heads, num_classes)
            else:
                self.heads = nn.ModuleList()
                for _ in range(num_heads):
                    self.heads.append(nn.Linear(embed_dims, num_attrs))
                self.rel = nn.Linear(num_attrs*num_heads, num_classes)
        else:
            self.attrs = modules.Local_Linear(embed_dims, num_attrs*num_heads, groups=num_heads)
            if self.use_graph:
                self.rel = modules.Generate_Attr_Relation_Graph(num_heads, num_attrs, num_classes, residual_graph=residual_graph)
            else:
                self.rel = nn.Linear(num_attrs*num_heads, num_classes)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_in = fan // m.groups
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}
    
    def forward_features(self, x):
        B = x.shape[0]
        x_ori = x
        x = self.patch_embed(x)
        x += self.pos_embeddings
        for idx, blk in enumerate(self.block):
            x = blk(x)
        return x

    def forward(self, x):
        dev = x.device
        x = self.forward_features(x)
        if not (self.att_type == "mbatt" or self.att_type == "fmbatt" or self.att_type == "mattratt"):
            x = torch.mean(x, dim=1)
            if self.repeat_emb:
                x = torch.split(x, self.embed_dims//self.num_heads, dim=1)
                ds_feature = []
                for x_split, head in zip(x, self.heads):
                    ds_feature.append(head(x_split).unsqueeze(1))
                ds_feature = torch.cat(ds_feature, dim=1)
                x = self.rel(ds_feature.view(ds_feature.shape[0], -1))
            else:
                ds_feature = []
                for head in self.heads:
                    ds_feature.append(head(x).unsqueeze(1))
                ds_feature = torch.cat(ds_feature, dim=1)
                x = self.rel(ds_feature.view(ds_feature.shape[0], -1))
            # print(x.shape)
            if self.use_DS:
                return x, ds_feature
        else:
            ds_feature = self.attrs(x.view(x.shape[0], -1)).permute(0, 2, 1).contiguous()    # B, num_attr, num_head
            # x = torch.mean(x, dim=-1).squeeze(-1)
            if self.use_mask:
                if not self.mask.device == dev:
                    self.mask = self.mask.to(dev)
                x = x * self.mask
            if self.use_graph:
                x, other_attr = self.rel(ds_feature)
                if self.use_DS:
                    return x, ds_feature
            else:
                x = self.rel(ds_feature.view(x.shape[0], -1))
                if self.use_DS:
                    return x, ds_feature
        return x
    

@register_model
def vit(pretrained=False, **kwargs):
    model = MG_Memorybank_ATT_ViT(patch_size=16, embed_dims=768,
                 num_heads=12, mlp_ratios=4., depths=12, pemb_type="pemb", use_IRB=True, att_type="att", use_localfc=False, use_proj=True, repeat_emb=False, eda_type="no", **kwargs)
    return model

@register_model
def vit_ll_noDS(pretrained=False, group_num=1, **kwargs):
    model = MG_Memorybank_ATT_ViT(patch_size=16, embed_dims=768,
                 num_heads=12, mlp_ratios=4., depths=12, pemb_type="pemb", use_IRB=True, att_type="att", use_localfc=True, use_proj=True, repeat_emb=False, eda_type="no", group_num=group_num, **kwargs)
    return model

@register_model
def vit_DS(pretrained=False, **kwargs):
    model = MG_Memorybank_ATT_ViT(patch_size=16, embed_dims=763,
                 num_heads=7, mlp_ratios=4., depths=12, pemb_type="pemb", use_IRB=True, att_type="att", use_localfc=False, use_proj=True, repeat_emb=False, eda_type="no", use_DS=True,**kwargs)
    return model

@register_model
def mgvit_DS(pretrained=False, **kwargs):
    model = MG_Memorybank_ATT_ViT(patch_size=16, embed_dims=763,
                 num_heads=7, mlp_ratios=4., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="att", use_localfc=False, use_proj=True, repeat_emb=False, eda_type="no", use_DS=True, **kwargs)
    return model

@register_model
def mgvit_ll_DS(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=7, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="att", use_localfc=True, use_proj=True, repeat_emb=True, local_fc_gratio=1, use_DS=True, eda_type="no", **kwargs)
    return model

@register_model
def mgmbatt_noEDA(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=7, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, use_proj=False, repeat_emb=True, local_fc_gratio=1, use_DS=True, eda_type="no", **kwargs)
    return model

@register_model
def mgmbatt_noDS(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=num_heads, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, local_fc_gratio=1, use_proj=False, repeat_emb=True, use_DS=False, eda_type="num_3", **kwargs)
    return model

@register_model
def mgmbatt(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=num_heads, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, local_fc_gratio=1, use_proj=False, repeat_emb=True, use_DS=True, eda_type="num_3", **kwargs)
    return model

@register_model
def mgmbatt_graph_noDS(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=8, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=num_heads, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, local_fc_gratio=1, use_proj=False, repeat_emb=True, use_graph=True, use_DS=False, eda_type="num_3", **kwargs)
    return model

# Ours
@register_model
def mgmbatt_graph(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=num_heads, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, local_fc_gratio=1, use_proj=False, repeat_emb=True, use_graph=True, use_DS=True, eda_type="num_3", residual_graph=False, **kwargs)
    return model


# MGPE repeat dim实验, 使用LL时不对embedding进行重复
@register_model
def mgvit_ll_DS_noRepeat(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=7, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="att", use_localfc=True, use_proj=True, repeat_emb=False, local_fc_gratio=1, use_DS=True, eda_type="no", **kwargs)
    return model

# LL分组数量实验, 无深度监督, 对768维的Embedding进行分组, 分组数量需要手动在modules里调节
@register_model
def vit_ll_group(pretrained=False, **kwargs):
    model = MG_Memorybank_ATT_ViT(patch_size=16, embed_dims=768,
                 num_heads=12, mlp_ratios=4., depths=12, pemb_type="pemb", use_IRB=True, att_type="att", use_localfc=True, use_proj=True, repeat_emb=False, eda_type="no", **kwargs)
    return model

# LL分组数量实验, 使用MGPE, 有深度监督, 对2688维的Embedding进行分组, 分组倍数通过local_fc_gratio参数调节
@register_model
def mgvit_ll_DS_heads_gratio(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, local_fc_gratio=1,**kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=7, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="att", use_localfc=True, use_proj=True, repeat_emb=True, local_fc_gratio=local_fc_gratio, use_DS=True, eda_type="no", **kwargs)
    return model

# MBEDA数量实验, 按照从后往前的顺序替换部分FFN为MBEDA
@register_model
def mgmbatt_partEDA(pretrained=False, num_heads=7, embdim_per_head=192, patch_size=32, eda_type="", **kwargs):
    embed_dims = embdim_per_head * num_heads
    model = MG_Memorybank_ATT_ViT(patch_size=patch_size, embed_dims=embed_dims,
                 num_heads=7, mlp_ratios=1., depths=12, pemb_type="mgpemb", use_IRB=True, att_type="mbatt", use_localfc=True, use_proj=False, repeat_emb=True, local_fc_gratio=1, use_DS=True, eda_type=eda_type, **kwargs)
    return model

if __name__ == "__main__":
    model = mgmbatt_graph(False, num_attrs=2, num_classes=3, patch_size=6, embdim_per_head=96, img_size=64, num_heads=11)
    print(model)
    model.train()
    img = torch.randn([1, 1, 64, 64])
    out = model(img)[0]
    print(out.shape)
    # print(out[1].shape)
    from thop import profile, clever_format
    with torch.no_grad():
        flops, params = profile(model, inputs=(img,))
        flops, params = clever_format([flops, params], "%.6f")
    print("Flops: ", flops)
    print("num_paras: ", params)