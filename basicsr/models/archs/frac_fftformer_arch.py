#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:09:07 2024

@author: dl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from basicsr.models.archs.frft import FRFT2D, IFRFT2D
# from frft import FRFT2D, IFRFT2D
import math
from torchvision import ops

#Modulated deformable convolution definition
class ModDeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, dilation=1, modulation=True):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(ModDeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding = padding, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride, dilation=dilation)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride, dilation=dilation)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))
            if self.dilation ==0:
                offset = torch.zeros_like(m.repeat(1, 2, 1, 1))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset

class DeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=True, modulation=False, dilation=1):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(1)

        # self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
        #                         padding=dilation, bias=bias)

        # self.p_conv.weight.data.zero_()
        # if bias:
        #     self.p_conv.bias.data.zero_()

        self.modulation = modulation
        if modulation:
            # self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, stride=stride, dilation=dilation,
            #                         padding=dilation, bias=bias)

            # self.m_conv.weight.data.zero_()
            # if bias:
            #     self.m_conv.bias.data.zero_()

            self.dconv = ModDeformConv2d(inc, outc, kernel_size, padding=padding, stride=stride, dilation=dilation, modulation=True)
        else:
            self.dconv = ModDeformConv2d(inc, outc, kernel_size, padding=padding, stride=stride, dilation =dilation, modulation=False)#DeformConv2d(inc, outc, kernel_size, padding=padding)

    def forward(self, x):

        if self.modulation:
            x_offset_conv = self.dconv(x)
        else:
            x_offset_conv = self.dconv(x)

        return x_offset_conv

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, activation=nn.SELU(inplace=True),
                 norm_layer=nn.InstanceNorm2d):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1),
                                   activation,
                                   nn.Conv2d(out_channels, in_channels, kernel_size, 1, 1))

    def forward(self, x):
        x = x + self.model(x)

        return x

class hallucination_module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1, padding=1, norm_layer=nn.InstanceNorm2d):
        super(hallucination_module, self).__init__()

        self.dilation = dilation

        if self.dilation != 0:

            self.hallucination_conv = DeformConv(out_channels, out_channels, padding = padding, modulation=True, dilation=self.dilation)

        else:

            self.m_conv = nn.Conv2d(in_channels, 3 * 3, kernel_size=3, stride=1, bias=True)

            self.m_conv.weight.data.zero_()
            self.m_conv.bias.data.zero_()

            self.dconv = ModDeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        if self.dilation != 0:

            hallucination_output = self.hallucination_conv(x)

        else:
            # hallucination_map = 0

            # mask = torch.sigmoid(self.m_conv(x))

            # offset = torch.zeros_like(mask.repeat(1, 2, 1, 1))

            hallucination_output = self.dconv(x)

        return hallucination_output#, hallucination_map

class hallucination_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilations=(0, 1, 2, 4), norm_layer=nn.InstanceNorm2d):
        super(hallucination_res_block, self).__init__()

        self.dilations = dilations

        self.hallucination_d0 = hallucination_module(in_channels, out_channels, dilations[0])
        self.hallucination_d1 = hallucination_module(in_channels, out_channels, dilations[1])
        self.hallucination_d2 = hallucination_module(in_channels, out_channels, dilations[2], padding=3)
        self.hallucination_d3 = hallucination_module(in_channels, out_channels, dilations[3], padding=9)

        self.mask_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                                       nn.SELU(inplace=True),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       ResBlock(out_channels, out_channels,
                                                norm_layer=norm_layer),
                                       nn.Conv2d(out_channels, 4, 1, 1))

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x):
        res = x

        d0_out = self.hallucination_d0(res)
        d1_out = self.hallucination_d1(res)
        d2_out = self.hallucination_d2(res)
        d3_out= self.hallucination_d3(res)

        mask = self.mask_conv(x)
        mask = torch.softmax(mask, 1)

        sum_out = d0_out * mask[:, 0:1, :, :] + d1_out * mask[:, 1:2, :, :] + \
                  d2_out * mask[:, 2:3, :, :] + d3_out * mask[:, 3:4, :, :]

        res = self.fusion_conv(sum_out)
        
        return res


#FRFT definition
class FRFT_layer(nn.Module):
    def __init__(self, in_channels, order=0.5):
        super(FRFT_layer, self).__init__()
        C0 = int(in_channels)
        C1 = int(in_channels)# - 2*C0
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv_05 = nn.Conv2d(2*C1, 2*C1, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(2*C0, 2*C0, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels*3, in_channels, kernel_size=3, padding=1)
        self.order = order
        self.aspdc_block = hallucination_res_block(in_channels, in_channels)

    def forward(self, x):
        N, C, H, W = x.shape

        C0 = int(C/3)
        x_0 = x#[:, 0:C0, :, :]
        x_05 = x#[:, C0:C-C0, :, :]
        x_1 = x#[:, C-C0:C, :, :]

        # order = 0
        x_0 = self.conv_0(x_0)
        x_0 = nn.LeakyReLU(inplace=True)(x_0)
        x_0 = self.aspdc_block(x_0)
        x_0 = nn.LeakyReLU(inplace=True)(x_0)
        x_0 = self.conv_0(x_0)
        
        # order = 0.5
        Fre = FRFT2D(x_05, self.order)
        Real = Fre.real
        Imag = Fre.imag
        Mix = torch.concat((Real, Imag), dim=1)
        Mix = self.conv_05(Mix)
        Mix = nn.LeakyReLU(inplace=True)(Mix)
        Mix = self.conv_05(Mix)
        Real1, Imag1 = torch.chunk(Mix, 2, 1)
        Fre_out = torch.complex(Real1, Imag1)
        IFRFT = IFRFT2D(Fre_out, self.order)
        IFRFT = torch.abs(IFRFT)#/(H*W)

        # order = 1
        fre = torch.fft.rfft2(x_1, norm='backward')
        real = fre.real
        imag = fre.imag
        mix = torch.concat((real, imag), dim=1)
        mix = self.conv_1(mix)
        mix = nn.LeakyReLU(inplace=True)(mix)
        mix = self.conv_1(mix)
        real1, imag1 = torch.chunk(mix, 2, 1)
        fre_out = torch.complex(real1, imag1)
        x_1 = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        output = torch.cat([x_0, IFRFT, x_1], dim=1)
        output = self.conv2(output)

        return output + x


#shallow_layer
class shallow_layer(nn.Module):
    def __init__(self, inchannel, out_channels):
        super(shallow_layer, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.elu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.elu(self.conv3(self.conv2(x))) + x
        x2 = self.elu(self.conv3(self.conv2(x1))) + x + x1
        
        return x2

#psf estimation block
class PEB(nn.Module):
    def __init__(self, inchannel, outchannel, patch_size=13):
        super(PEB, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, outchannel, 1, stride = 1, padding=0)
        self.gelu = nn.LeakyReLU(inplace=True)
        self.dilation1 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=1, dilation = 1)
        
        self.dilation2 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=3, dilation = 3)
        self.dilation3 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=5, dilation = 5)
        
        self.conv2 = nn.Conv2d(outchannel*3, patch_size * patch_size, 1, stride = 1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.layernorm = nn.BatchNorm2d(patch_size * patch_size)
        self.softmax = nn.Softmax(dim=1)
        self.patch_size = patch_size

    def forward(self,x):
        batch, channel, height, width = x.size()
        y = self.conv1(x)
        y1 = self.dilation1(y)
        y1 = self.gelu(y1)
        
        y2 = self.dilation2(y)
        y2 = self.gelu(y2)
        
        y3 = self.dilation3(y)
        y3 = self.gelu(y3)

#        y1 = self.dilation1(y)
        conc = torch.cat([y1,y2,y3], dim=1)
        out = self.avg_pool(conc)
        out = self.conv2(out)
        out = self.layernorm(out)
        out = self.softmax(out)
        out = out.view(batch, 1, self.patch_size, self.patch_size) 
        # for jj in range(out.shape[0]):
        #     out[jj:jj+1,:,:,:] = torch.div(out[jj:jj+1,:,:,:], torch.sum(out[jj:jj+1,:,:,:]))
        return out

class PEB2(nn.Module):
    def __init__(self, inchannel, outchannel, patch_size=13):
        super(PEB2, self).__init__()
        
        self.conv1 = nn.Conv2d(inchannel, outchannel, 1, stride = 1, padding=0)
        self.gelu = nn.LeakyReLU(inplace=True)
        self.dilation1 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=1, dilation = 1)
        
        self.dilation2 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=3, dilation = 3)
        self.dilation3 = nn.Conv2d(outchannel, outchannel, 3, stride = 1, padding=5, dilation = 5)
        
        self.conv2 = nn.Conv2d(outchannel*3, patch_size * patch_size, 1, stride = 1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.layernorm = nn.BatchNorm2d(patch_size * patch_size)
        self.softmax = nn.Softmax(dim=1)
        self.patch_size = patch_size

    def forward(self,x):
        batch, channel, height, width = x.size()
        y = self.conv1(x)
        y1 = self.dilation1(y)
        y1 = self.gelu(y1)
        
        y2 = self.dilation2(y)
        y2 = self.gelu(y2)
        
        y3 = self.dilation3(y)
        y3 = self.gelu(y3)

#        y1 = self.dilation1(y)
        conc = torch.cat([y1,y2,y3], dim=1)
        out = self.avg_pool(conc)
        out = self.conv2(out)
        out = self.layernorm(out)
        out = self.softmax(out)
        out = out.view(batch, 1, self.patch_size, self.patch_size) 
        # for jj in range(out.shape[0]):
        #     out[jj:jj+1,:,:,:] = torch.div(out[jj:jj+1,:,:,:], torch.sum(out[jj:jj+1,:,:,:]))
        return out

#fractional weiner
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class fractional_weiner_FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(fractional_weiner_FFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size//2 +1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class fractional_deformable_SA(nn.Module):
    def __init__(self, dim, bias):
        super(fractional_deformable_SA, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        
        # b,c,h,w, patch1_size, patch2_size = q_patch.size()
        
        q_fft = FRFT2D(q_patch.float(), 0.5)#torch.zeros(q_patch.size(), dtype=torch.complex128).cuda()
        k_fft = FRFT2D(k_patch.float(), 0.5)#torch.zeros(q_patch.size(), dtype=torch.complex128).cuda()
                
        out = q_fft * k_fft
        out = torch.abs(IFRFT2D(out, 0.5)) #torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        # out = torch.abs(out)/(self.patch_size*self.patch_size)
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out    #nn.Softmax()(out)
        output = self.project_out(output)
        return output
# class fractional_deformable_SA(nn.Module):
#     def __init__(self, dim, bias):
#         super(fractional_deformable_SA, self).__init__()

#         self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
#         self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

#         self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

#         self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

#         self.patch_size = 16
        

#     def forward(self, x):
#         hidden = self.to_hidden(x)

#         q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

#         b, c, h, w = q.size()
        
#         q_fft = FRFT2D(q, 0.5)#torch.fft.rfft2(q_patch.float())
#         k_fft = FRFT2D(k, 0.5)#torch.fft.rfft2(k_patch.float())
    
                
#         out = q_fft * k_fft
#         out = IFRFT2D(out, 0.5)#torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
#         out = torch.abs(out)/(h*w)#(self.patch_size*self.patch_size)
#         # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
#         #                 patch2=self.patch_size)

#         out = self.norm(out)

#         output = v * out
#         output = self.project_out(output)
#         return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()

        
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = fractional_deformable_SA(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = fractional_weiner_FFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        x = nn.LeakyReLU(inplace=True)(x)
        return x




class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),#nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                 # nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(#nn.ConvTranspose2d(n_feat, n_feat, kernel_size=2, stride=2),#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat* 2, 3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
## Resizing modules
#class Downsample(nn.Module):
#    def __init__(self, n_feat):
#        super(Downsample, self).__init__()

#        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=4, stride=2, padding=1),#nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
#                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False),
#                                  nn.SELU())
#
#    def forward(self, x):
#        return self.body(x)


#class Upsample(nn.Module):
#    def __init__(self, n_feat):
#        super(Upsample, self).__init__()

#        self.body = nn.Sequential(nn.ConvTranspose2d(n_feat, n_feat, kernel_size=2, stride=2),#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False),
#                                  nn.SELU())

#    def forward(self, x):
 #       return self.body(x)


##########################################################################
##---------- FFTformer -----------------------

class frac_fftformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4,8,12,12,8,4],#,
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 dual_pixel_task = False
                 ):
        super(frac_fftformer, self).__init__()
        
        self.kernel_estimator = PEB(inp_channels,64)
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.shallow_layer1 = shallow_layer(dim, dim)
        self.shallow_layer2 = shallow_layer(dim, dim*2)
        self.shallow_layer3 = shallow_layer(dim, dim*4)
        
        self.frft_layer1 = FRFT_layer(dim)
        self.frft_layer2 = FRFT_layer(dim*2)
        self.frft_layer3 = FRFT_layer(dim*4)
        
        
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) for i in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[3])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[4])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_blocks[5])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias) for i in range(num_refinement_blocks)])

        # self.fuse2 = Fuse(dim * 2)
        # self.fuse1 = Fuse(dim)
        self.output1 = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output2 = nn.Conv2d(int(dim*2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output3 = nn.Conv2d(int(dim*4), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        
        self.conv1 = nn.Sequential(nn.Conv2d(dim*4, dim*2, kernel_size=1, padding=0))
        self.conv2 = nn.Sequential(nn.Conv2d(dim*8, dim*4, kernel_size=1, padding=0))
        self.conv3 = nn.Sequential(nn.Conv2d(dim*2, dim, kernel_size=1, padding=0))
        
        self.act = nn.LeakyReLU(inplace=True)
        
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv1 = nn.Conv2d(dim, int(dim*1**1), kernel_size=1, bias=bias)
            self.skip_conv2 = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
            self.skip_conv3 = nn.Conv2d(dim, int(dim*4**1), kernel_size=1, bias=bias)
        
        self.kernel_shape = 13
        # self.kernel = nn.Parameter(torch.rand((, 1, 1, self.patch_size, self.patch_size//2 +1)))

    def forward(self, inp_img):
        batch, c, h, w = inp_img.size()
        inp_img2= F.interpolate(inp_img, scale_factor=0.5)
        inp_img4= F.interpolate(inp_img2, scale_factor=0.5)
        
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level11 = self.shallow_layer1(inp_enc_level1)
        out_enc_level1 = self.frft_layer1(inp_enc_level11)
        out_enc_level1 = self.encoder_level1(out_enc_level1)
        
        inp_enc_level2_ini = self.patch_embed(inp_img2)
        inp_enc_level2 = torch.cat([self.shallow_layer2(inp_enc_level2_ini), self.down1_2(out_enc_level1)], dim=1)
        inp_enc_level22 = self.conv1(inp_enc_level2)
        out_enc_level2 = self.frft_layer2(inp_enc_level22)
        out_enc_level2 = self.encoder_level2(out_enc_level2)
        
        inp_enc_level3_ini = self.patch_embed(inp_img4)
        inp_enc_level3 = torch.cat([self.shallow_layer3( inp_enc_level3_ini), self.down2_3(out_enc_level2)], dim=1)
        inp_enc_level33 = self.conv2(inp_enc_level3)
        out_enc_level3 = self.frft_layer3(inp_enc_level33)
        out_enc_level3 = self.encoder_level3(out_enc_level3)
        
        
        inp_dec_level3 = self.frft_layer3(out_enc_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) # 1st output from here
        if self.dual_pixel_task:
            out_dec_level3_res = out_dec_level3 + self.skip_conv3(inp_enc_level3_ini)
            out_img3 = self.output3(out_dec_level3_res)
        else:
            out_img3 = inp_img4 + self.output3(out_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.conv1(torch.cat((out_enc_level2, inp_dec_level2), dim =1))
        inp_dec_level2 = self.frft_layer2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) # 2nd output from here
        if self.dual_pixel_task:
            out_dec_level2_res = out_dec_level2 + self.skip_conv2(inp_enc_level2_ini)
            out_img2 = self.output2(out_dec_level2_res)
        else:
            out_img2 = inp_img2 + self.output2(out_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.conv3(torch.cat((out_enc_level1, inp_dec_level1), dim =1))
        inp_dec_level1 = self.frft_layer1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # final output from here
        out_dec_level1 = self.refinement(out_dec_level1)
        
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv1(inp_enc_level1)
            out_img = self.output1(out_dec_level1)
        else:        
            out_img = inp_img + self.output1(out_dec_level1)
        
        return out_img, out_img2, out_img3


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            layers.append(nn.Conv2d(out_filters, out_filters, 3, 2, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 48, normalization=False),
            *discriminator_block(48, 96),
            *discriminator_block(96, 96*2),
            *discriminator_block(96*2, 96*4),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(96*4, 1, 1, padding=0, bias=False)
        )

    def forward(self, img_A):
        # Concatenate image and condition image by channels to produce input
        # img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_A)


if __name__ == "__main__":
    from torchsummary import summary
    ngpu = 1
    device= torch.device("cuda:0")
    x = torch.rand((2,3,256,256)).to(device)
    model = frac_fftformer().to(device)
    # out = model(x)
    summary(model, (3, 256, 256))


