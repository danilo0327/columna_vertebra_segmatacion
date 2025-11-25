# Modelo de segmentación DeepLabV3+
import os
import json
import zipfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision.models import resnet50

from ..config import (
    MODEL_ZIP_PATH,
    CLASSES_JSON_PATH,
    MODEL_EXTRACTED_DIR,
    INPUT_SIZE,
    NUM_CLASSES,
    AVAILABLE_MODELS
)


# ============================================================================
# MÓDULOS BASE DE LA ARQUITECTURA
# ============================================================================

class ConvBlock(nn.Module):
    """Bloque convolucional con BatchNorm y ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ConvBlockWithDropout(nn.Module):
    """Bloque convolucional con BatchNorm, ReLU y Dropout (para DeepLabV3++)"""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ConvBlockExtended(nn.Module):
    """Bloque convolucional extendido (para DeepLabV3pp) - tiene más capas"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Estructura según checkpoint: Conv(0) -> BN(1) -> ReLU -> Conv(4) -> BN(5) -> ReLU
        # Índice 3 no existe en el modelo guardado
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),  # índice 0
            nn.BatchNorm2d(out_ch),  # índice 1
            nn.ReLU(inplace=True),
            # Índice 3 no existe - dejar espacio
            nn.Identity(),  # Placeholder para índice 3
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),  # índice 4
            nn.BatchNorm2d(out_ch),  # índice 5
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = F.interpolate(self.pool(x), size=size, mode='bilinear', align_corners=True)
        out = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        return self.project(out)


class ChannelAttention(nn.Module):
    """Channel Attention para ASPP"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class ASPPWithAttention(nn.Module):
    """ASPP mejorado con Channel Attention (según código original)"""
    def __init__(self, in_ch, out_ch, rates=[6, 12, 18]):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Atrous blocks
        self.atrous_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for r in rates
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention
        total_ch = out_ch * (len(rates) + 2)  # conv1 + atrous_blocks + global_pool
        self.channel_attention = ChannelAttention(total_ch, reduction=16)
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        feats = [self.conv1(x)] + [block(x) for block in self.atrous_blocks]
        feats.append(F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=True))
        feats = torch.cat(feats, dim=1)
        feats = self.channel_attention(feats)
        return self.project(feats)


class DeepLabV3Plus(nn.Module):
    """Arquitectura DeepLabV3+ personalizada"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.aspp = ASPP(512, 256)
        
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.out = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x):
        size = x.shape[2:]
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        aspp_out = self.aspp(enc4)
        low_level = self.decoder_conv1(enc1)
        aspp_up = F.interpolate(aspp_out, size=low_level.shape[2:], mode='bilinear', align_corners=True)
        dec = torch.cat([aspp_up, low_level], dim=1)
        dec = self.decoder_conv2(dec)
        out = F.interpolate(dec, size=size, mode='bilinear', align_corners=True)
        
        return self.out(out)


class AttentionGate(nn.Module):
    """Attention Gate (según código original)"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        return x * self.psi(self.relu(g1 + x1))


class SpatialAttentionModule(nn.Module):
    """Spatial Attention (según código original)"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))


class DeepLabV3pp(nn.Module):
    """DeepLabV3++ - DeepLabV3+ con Decoder Denso tipo U-Net++ (según código original)"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        
        # Encoder (usar ConvBlockWithDropout según código original)
        self.enc1 = ConvBlockWithDropout(in_channels, 64, dropout=0.05)
        self.enc2 = ConvBlockWithDropout(64, 128, dropout=0.1)
        self.enc3 = ConvBlockWithDropout(128, 256, dropout=0.1)
        self.enc4 = ConvBlockWithDropout(256, 512, dropout=0.15)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # ASPP en bottleneck
        self.aspp = ASPPWithAttention(512, 256, rates=[6, 12, 18])
        
        # Projections (según código original)
        self.enc4_proj = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder DENSO (según código original)
        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.att4 = AttentionGate(256, 256, 128)
        self.dec4 = ConvBlockWithDropout(512, 256, dropout=0.1)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.enc3_proj = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = ConvBlockWithDropout(256, 128, dropout=0.1)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.enc2_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.spatial_att = SpatialAttentionModule()
        self.dec2 = ConvBlockWithDropout(128, 64, dropout=0.05)
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlockWithDropout(128, 64, dropout=0.05)
        
        # Output (según código original)
        self.out = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # ASPP bottleneck
        bottleneck = self.aspp(enc4)
        
        # Decoder denso
        dec4 = self.up4(bottleneck)
        enc4_reduced = self.enc4_proj(enc4)
        enc4_reduced = F.interpolate(enc4_reduced, size=dec4.shape[2:], 
                                    mode='bilinear', align_corners=True)
        enc4_att = self.att4(dec4, enc4_reduced)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        enc3_reduced = self.enc3_proj(enc3)
        enc3_reduced = F.interpolate(enc3_reduced, size=dec3.shape[2:], 
                                    mode='bilinear', align_corners=True)
        enc3_att = self.att3(dec3, enc3_reduced)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        enc2_reduced = self.enc2_proj(enc2)
        enc2_reduced = F.interpolate(enc2_reduced, size=dec2.shape[2:], 
                                    mode='bilinear', align_corners=True)
        enc2_att = self.spatial_att(enc2_reduced)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], 
                           mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        output = self.out(dec1)
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], 
                                 mode='bilinear', align_corners=True)
        
        return output


# ============================================================================
# MODELO: DeepLabV3PlusDenseDecoder (del notebook DeepLabV3+Densedecoder.ipynb)
# ============================================================================

class ASPP_ResNet(nn.Module):
    """ASPP para ResNet50 (2048 canales de entrada)"""
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[0],
                      dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[1],
                      dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rates[2],
                      dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=(h, w), mode="bilinear", align_corners=False)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(x_cat)


class DenseDecoderBlock(nn.Module):
    """Bloque de decoder denso con atención"""
    def __init__(self, in_ch, skip_ch, prev_ch, out_ch, use_attention=True):
        super().__init__()
        self.use_attention = use_attention and skip_ch > 0
        if self.use_attention:
            self.att = AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=out_ch)

        # conv para fusionar (in_ch + prev_ch + skip_ch) -> out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + prev_ch + (skip_ch if self.use_attention else 0),
                      out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None, prev_upsampled=None, scale_factor=2):
        x_up = F.interpolate(x, scale_factor=scale_factor, mode="bilinear",
                             align_corners=False)

        feats = [x_up]

        if prev_upsampled is not None:
            feats.append(prev_upsampled)

        if self.use_attention and skip is not None:
            # redimensionar skip al tamaño de x_up por si acaso
            if skip.shape[2:] != x_up.shape[2:]:
                skip = F.interpolate(skip, size=x_up.shape[2:], mode="bilinear",
                                     align_corners=False)
            skip_att = self.att(x_up, skip)
            feats.append(skip_att)

        x_cat = torch.cat(feats, dim=1)
        out = self.conv(x_cat)
        return out


class DeepLabV3PlusDenseDecoder(nn.Module):
    """DeepLabV3+ con ResNet50 backbone y decoder denso (del notebook)"""
    def __init__(self, num_classes=3):
        super().__init__()

        # ResNet50 backbone (como DeepLab, con dilataciones)
        self.backbone = resnet50(weights=None,
                                 replace_stride_with_dilation=[False, True, True])

        # Tomamos features en varios niveles
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
        )  # ~128x128 para input 256
        self.maxpool = self.backbone.maxpool          # 64x64
        self.layer1 = self.backbone.layer1            # 64x64
        self.layer2 = self.backbone.layer2            # 32x32
        self.layer3 = self.backbone.layer3            # 32x32 (dilatado)
        self.layer4 = self.backbone.layer4            # 32x32 (dilatado)

        # ASPP sobre layer4
        self.aspp = ASPP_ResNet(in_channels=2048, out_channels=256)

        # Proyección de skips
        self.enc_32 = nn.Conv2d(1024, 256, 1)   # de layer3 (32x32)
        self.enc_64 = nn.Conv2d(256, 128, 1)    # de layer1 (64x64)
        self.enc_128 = nn.Conv2d(64, 64, 1)     # de layer0 (128x128)

        # Decoder denso (4 niveles)
        # Nivel 0: 32x32 (no upsample todavía, solo conv)
        self.dec0_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Nivel 1: 32 -> 64, con skip de 64x64
        self.dec1 = DenseDecoderBlock(
            in_ch=256,   # viene de dec0
            skip_ch=128, # enc_64
            prev_ch=0,
            out_ch=128,
            use_attention=True,
        )

        # Nivel 2: 64 -> 128, con skip de 128x128 y conexiones densas
        self.dec2 = DenseDecoderBlock(
            in_ch=128,
            skip_ch=64,   # enc_128
            prev_ch=256,  # upsample de dec0
            out_ch=64,
            use_attention=True,
        )

        # Nivel 3: 128 -> 256, sin skip (solo dense de prev)
        self.dec3 = DenseDecoderBlock(
            in_ch=64,
            skip_ch=0,      # sin skip extra
            prev_ch=128+256, # upsample de dec1 y dec0
            out_ch=64,
            use_attention=False,
        )

        # Clasificador final
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)            # 128x128
        x0p = self.maxpool(x0)         # 64x64
        x1 = self.layer1(x0p)          # 64x64
        x2 = self.layer2(x1)           # 32x32
        x3 = self.layer3(x2)           # 32x32
        x4 = self.layer4(x3)           # 32x32

        # ASPP
        x_aspp = self.aspp(x4)         # 32x32

        # Skips proyectados
        enc32 = self.enc_32(x3)        # 256 canales, 32x32
        enc64 = self.enc_64(x1)        # 128 canales, 64x64
        enc128 = self.enc_128(x0)      # 64 canales, 128x128

        # Decoder nivel 0 (32x32)
        d0 = self.dec0_conv(x_aspp)    # 256, 32x32

        # Decoder nivel 1 (32 -> 64), con atención sobre enc64
        d1 = self.dec1(d0, skip=enc64, prev_upsampled=None, scale_factor=2)  # 128, 64x64

        # Decoder nivel 2 (64 -> 128), denso: usa d1 y d0 upsamplados + skip enc128
        d0_up_128 = F.interpolate(d0, scale_factor=4, mode="bilinear", align_corners=False)  # 256, 128x128
        d2 = self.dec2(d1, skip=enc128, prev_upsampled=d0_up_128, scale_factor=2)           # 64, 128x128

        # Decoder nivel 3 (128 -> 256), denso: concat de d2 + d1_up + d0_up
        d1_up_256 = F.interpolate(d1, scale_factor=4, mode="bilinear", align_corners=False) # 128, 256x256
        d0_up_256 = F.interpolate(d0, scale_factor=8, mode="bilinear", align_corners=False) # 256, 256x256

        # Para el bloque dec3, pasamos una "prev_upsampled" que concatena d1_up y d0_up
        prev_cat = torch.cat([d1_up_256, d0_up_256], dim=1)  # (128+256, 256x256)
        d3 = self.dec3(d2, skip=None, prev_upsampled=prev_cat, scale_factor=2)               # 64, 256x256

        logits = self.classifier(d3)   # (B, num_classes, 256,256)
        return logits


class UNetPlusPlus(nn.Module):
    """Arquitectura U-Net++ personalizada"""
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv0_0 = ConvBlock(in_channels, 64)
        self.conv1_0 = ConvBlock(64, 128)
        self.conv2_0 = ConvBlock(128, 256)
        self.conv3_0 = ConvBlock(256, 512)
        self.conv4_0 = ConvBlock(512, 1024)
        
        self.conv0_1 = ConvBlock(64 + 128, 64)
        self.conv1_1 = ConvBlock(128 + 256, 128)
        self.conv2_1 = ConvBlock(256 + 512, 256)
        self.conv3_1 = ConvBlock(512 + 1024, 512)
        
        self.conv0_2 = ConvBlock(64 * 2 + 128, 64)
        self.conv1_2 = ConvBlock(128 * 2 + 256, 128)
        self.conv2_2 = ConvBlock(256 * 2 + 512, 256)
        
        self.conv0_3 = ConvBlock(64 * 3 + 128, 64)
        self.conv1_3 = ConvBlock(128 * 3 + 256, 128)
        
        self.conv0_4 = ConvBlock(64 * 4 + 128, 64)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        return self.out(x0_4)


class SegmentationModel:
    """Clase para cargar y usar modelos de segmentación"""
    
    def __init__(self, model_type: str = "deeplabv3plus"):
        """
        Inicializa el modelo de segmentación
        
        Args:
            model_type: Tipo de modelo a usar ('deeplabv3plus' o 'unetplusplus')
        """
        if model_type not in AVAILABLE_MODELS:
            raise ValueError(f"Modelo no disponible: {model_type}. Modelos disponibles: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_config = AVAILABLE_MODELS[model_type]
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self._load_classes()
        self.model_loaded = False
        
    def _load_classes(self) -> list:
        """Carga las clases desde el archivo JSON del modelo"""
        try:
            classes_path = self.model_config["model_dir"] / self.model_config["classes_file"]
            if not classes_path.exists():
                # Fallback al archivo principal
                classes_path = CLASSES_JSON_PATH
            
            with open(classes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Manejar diferentes formatos de JSON
            if isinstance(data, list):
                # Formato simple: ["Background", "T1", "V"]
                return data
            elif isinstance(data, dict):
                # Formato con id2label: {"id2label": {"0": "F", "1": "V", ...}}
                if "id2label" in data:
                    # Ordenar por id y extraer labels
                    id2label = data["id2label"]
                    num_classes = len(id2label)
                    classes = [id2label[str(i)] for i in range(num_classes)]
                    return classes
                elif "classes" in data:
                    return data["classes"]
                else:
                    # Si es un dict pero no tiene el formato esperado, intentar como lista
                    return list(data.values()) if data else ["Background", "T1", "V"]
            else:
                return ["Background", "T1", "V"]
        except Exception as e:
            print(f"Error cargando clases: {e}")
            return ["Background", "T1", "V"]
    
    def _find_model_file(self) -> Optional[Path]:
        """Busca el archivo del modelo"""
        model_dir = self.model_config["model_dir"]
        model_file = model_dir / self.model_config["model_file"]
        
        if model_file.exists():
            return model_file
        
        # Buscar cualquier archivo .pth en el directorio
        for file_path in model_dir.glob("*.pth"):
            if file_path.is_file():
                return file_path
        
        return None
    
    def load_model(self):
        """Carga el modelo de segmentación"""
        if self.model_loaded:
            return
        
        try:
            model_path = self._find_model_file()
            
            if model_path is None:
                model_dir = self.model_config["model_dir"]
                raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_dir}")
            
            print(f"Cargando modelo {self.model_config['name']} desde {model_path}...")
            
            # Intentar cargar como modelo PyTorch
            try:
                # PyTorch 2.6+ cambió el default de weights_only a True por seguridad
                # Como estos son nuestros modelos propios y confiables, usamos weights_only=False
                # También agregamos safe globals para numpy arrays que pueden estar en el checkpoint
                try:
                    # Intentar agregar safe globals para numpy (si está disponible en la versión)
                    if hasattr(torch.serialization, 'add_safe_globals'):
                        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                except (AttributeError, ImportError):
                    # Si no está disponible, continuamos sin ello
                    pass
                
                # Cargar checkpoint con weights_only=False (confiamos en nuestros modelos)
                # Usar map_location='cpu' primero para evitar problemas de memoria con GPU
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                    print(f"Checkpoint cargado exitosamente desde {model_path}")
                except EOFError as e:
                    raise RuntimeError(
                        f"Error al cargar el modelo: El archivo parece estar corrupto o incompleto.\n"
                        f"Archivo: {model_path}\n"
                        f"Error: {str(e)}\n"
                        f"Posibles soluciones:\n"
                        f"1. Verifica que el archivo se descargó completamente desde Git LFS\n"
                        f"2. Ejecuta: git lfs pull\n"
                        f"3. Verifica el tamaño del archivo (debe ser > 100 MB para modelos U-Net++)"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"Error inesperado al cargar el modelo desde {model_path}: {str(e)}"
                    ) from e
                
                # Intentar diferentes estructuras de checkpoint
                if isinstance(checkpoint, dict):
                    # Verificar si el checkpoint es directamente un state_dict (OrderedDict)
                    first_key = list(checkpoint.keys())[0] if checkpoint else ""
                    
                    # Verificar primero si es un state_dict directo (sin wrapper)
                    # Esto debe verificarse ANTES de buscar 'model' o 'model_state_dict'
                    is_direct_state_dict = (
                        first_key.startswith("enc1") or 
                        first_key.startswith("conv0_0") or
                        first_key.startswith("backbone") or
                        (not first_key.startswith("model.") and 
                         'model_state_dict' not in checkpoint and 
                         'state_dict' not in checkpoint and 
                         'model' not in checkpoint and
                         len(checkpoint) > 10)  # Si tiene muchas keys, probablemente es un state_dict
                    )
                    
                    # Caso 1: Checkpoint es directamente un state_dict con prefijo "model."
                    if first_key.startswith("model.") and 'model_state_dict' not in checkpoint and 'state_dict' not in checkpoint:
                        # El checkpoint es directamente el state_dict con prefijo "model."
                        # Necesitamos remover el prefijo o cargar el modelo completo
                        print("Detectado checkpoint con prefijo 'model.' - cargando modelo de torchvision...")
                        try:
                            from torchvision.models.segmentation import deeplabv3_resnet50
                            self.model = deeplabv3_resnet50(
                                num_classes=NUM_CLASSES,
                                pretrained_backbone=False
                            )
                            # Remover el prefijo "model." de las keys
                            state_dict_clean = {}
                            for key, value in checkpoint.items():
                                new_key = key.replace("model.", "", 1) if key.startswith("model.") else key
                                state_dict_clean[new_key] = value
                            self.model.load_state_dict(state_dict_clean, strict=False)
                            self.model = self.model.to(self.device)
                            self.model.eval()
                            print(f"Modelo {self.model_config['name']} cargado exitosamente")
                            self.model_loaded = True
                            print(f"Modelo cargado exitosamente en {self.device}")
                            return
                        except Exception as e:
                            print(f"Error cargando modelo con prefijo 'model.': {e}")
                            # Continuar con el flujo normal
                    
                    # Caso 2: Checkpoint es directamente un state_dict sin wrapper (como DeepLabV3pp)
                    elif is_direct_state_dict:
                        # Es un state_dict directo de una arquitectura personalizada
                        state_dict = checkpoint
                        architecture_name = self.model_config["architecture"]
                        
                        # Detectar arquitectura por las keys
                        has_backbone = 'backbone' in first_key  # DeepLabV3PlusDenseDecoder tiene backbone
                        has_dec1 = any('dec1' in k for k in list(checkpoint.keys())[:30])
                        has_att3 = any('att3' in k for k in list(checkpoint.keys())[:30])
                        is_deeplabpp = has_dec1 and has_att3 and not has_backbone  # DeepLabV3pp tiene decoder denso y atención pero NO backbone
                        is_deeplab_dense = has_backbone  # DeepLabV3PlusDenseDecoder tiene backbone
                        is_deeplab = 'enc1' in first_key and ('aspp' in first_key or 'decoder' in first_key or any('decoder' in k for k in list(checkpoint.keys())[:20]))
                        is_unetpp = 'conv0_0' in first_key or 'conv0_1' in first_key
                        
                        print(f"Detectado state_dict directo - Reconstruyendo arquitectura {architecture_name}...")
                        print(f"Primera key: {first_key}")
                        print(f"Total de parámetros: {len(state_dict)}")
                        print(f"Es DeepLabV3PlusDenseDecoder: {is_deeplab_dense} (backbone: {has_backbone})")
                        print(f"Es DeepLabV3pp: {is_deeplabpp} (dec1: {has_dec1}, att3: {has_att3})")
                        
                        if architecture_name == "DeepLabV3PlusDenseDecoder" or is_deeplab_dense:
                            try:
                                print("Construyendo arquitectura DeepLabV3PlusDenseDecoder...")
                                self.model = DeepLabV3PlusDenseDecoder(
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes (primeras 5: {missing_keys[:5]})")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas (primeras 5: {unexpected_keys[:5]})")
                                
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                                self.model_loaded = True
                                print(f"Modelo cargado exitosamente en {self.device}")
                                return
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3PlusDenseDecoder: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura.\n"
                                    f"Primera key: {first_key}"
                                )
                        elif architecture_name == "DeepLabV3pp" or is_deeplabpp:
                            try:
                                print("Construyendo arquitectura DeepLabV3pp...")
                                self.model = DeepLabV3pp(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes (primeras 5: {missing_keys[:5]})")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas (primeras 5: {unexpected_keys[:5]})")
                                
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                                self.model_loaded = True
                                print(f"Modelo cargado exitosamente en {self.device}")
                                return
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3pp: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura.\n"
                                    f"Primera key: {first_key}"
                                )
                        elif architecture_name == "DeepLabV3Plus" or is_deeplab:
                            try:
                                print("Construyendo arquitectura DeepLabV3Plus...")
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                
                                # Filtrar keys que no coinciden en tamaño para evitar errores
                                model_state_dict = self.model.state_dict()
                                filtered_state_dict = {}
                                skipped_keys = []
                                
                                for key, value in state_dict.items():
                                    if key in model_state_dict:
                                        model_shape = model_state_dict[key].shape
                                        checkpoint_shape = value.shape
                                        if model_shape == checkpoint_shape:
                                            filtered_state_dict[key] = value
                                        else:
                                            skipped_keys.append(f"{key}: checkpoint {checkpoint_shape} vs model {model_shape}")
                                    else:
                                        # Key no existe en el modelo, la omitimos
                                        skipped_keys.append(f"{key}: no existe en modelo")
                                
                                if skipped_keys:
                                    print(f"⚠️  Omitiendo {len(skipped_keys)} keys que no coinciden (primeras 5):")
                                    for skip in skipped_keys[:5]:
                                        print(f"   - {skip}")
                                
                                # Cargar solo las keys que coinciden
                                missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes en el modelo (primeras 5: {missing_keys[:5]})")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas (primeras 5: {unexpected_keys[:5]})")
                                
                                print(f"✅ Cargadas {len(filtered_state_dict)}/{len(state_dict)} keys del checkpoint")
                                
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                                self.model_loaded = True
                                print(f"Modelo cargado exitosamente en {self.device}")
                                return
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3+: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura.\n"
                                    f"Primera key: {first_key}"
                                )
                        elif architecture_name == "UNetPlusPlus" or is_unetpp:
                            try:
                                print(f"Construyendo arquitectura UNetPlusPlus...")
                                self.model = UNetPlusPlus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes (primeras 5: {missing_keys[:5]})")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas (primeras 5: {unexpected_keys[:5]})")
                                
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                                self.model_loaded = True
                                print(f"Modelo cargado exitosamente en {self.device}")
                                return
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo UNet++: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        else:
                            raise ValueError(f"Arquitectura '{architecture_name}' no soportada para state_dict directo.")
                    
                    # Caso 3: Checkpoint tiene wrapper con 'model', 'model_state_dict', etc.
                    elif 'model' in checkpoint:
                        # Modelo completo guardado
                        self.model = checkpoint['model']
                        if hasattr(self.model, 'eval'):
                            self.model.eval()
                    elif 'model_state_dict' in checkpoint:
                        # Solo state_dict - necesitamos reconstruir la arquitectura
                        state_dict = checkpoint['model_state_dict']
                        
                        # Determinar qué arquitectura usar
                        first_key = list(state_dict.keys())[0] if state_dict else ""
                        architecture_name = self.model_config["architecture"]
                        
                        # Detectar arquitectura por las keys del state_dict
                        is_deeplab = 'enc1' in first_key and 'aspp' in first_key and 'decoder_conv' in first_key
                        is_unetpp = 'conv0_0' in first_key or 'conv0_1' in first_key
                        
                        print(f"Reconstruyendo arquitectura {architecture_name}...")
                        
                        if architecture_name == "DeepLabV3Plus" or is_deeplab:
                            try:
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3+: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        elif architecture_name == "UNetPlusPlus" or is_unetpp:
                            try:
                                print(f"Construyendo arquitectura UNetPlusPlus...")
                                self.model = UNetPlusPlus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                # Intentar cargar con strict=False primero para ver qué falta
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes (primeras 5: {missing_keys[:5]})")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas (primeras 5: {unexpected_keys[:5]})")
                                
                                print(f"Moviendo modelo a {self.device}...")
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo UNet++: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura.\n"
                                    f"Primera key del state_dict: {list(state_dict.keys())[0] if state_dict else 'N/A'}"
                                )
                        else:
                            # Intentar con arquitecturas estándar de torchvision
                            print("Reconstruyendo arquitectura DeepLabV3+ estándar...")
                            # Intentar con ResNet50 primero (más común)
                            try:
                                from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
                                self.model = deeplabv3_resnet50(
                                    num_classes=NUM_CLASSES,
                                    pretrained_backbone=False
                                )
                                self.model.load_state_dict(state_dict, strict=False)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ ResNet50 cargado exitosamente")
                            except Exception as e1:
                                print(f"Error con ResNet50: {e1}")
                                print("Intentando con ResNet101...")
                                try:
                                    self.model = deeplabv3_resnet101(
                                        num_classes=NUM_CLASSES,
                                        pretrained_backbone=False
                                    )
                                    self.model.load_state_dict(state_dict, strict=False)
                                    self.model = self.model.to(self.device)
                                    self.model.eval()
                                    print("Modelo DeepLabV3+ ResNet101 cargado exitosamente")
                                except Exception as e2:
                                    raise RuntimeError(
                                        f"No se pudo cargar el modelo con ResNet50 ni ResNet101.\n"
                                        f"El modelo parece usar una arquitectura personalizada.\n"
                                        f"Error ResNet50: {str(e1)[:200]}\n"
                                        f"Error ResNet101: {str(e2)[:200]}"
                                    )
                    elif 'state_dict' in checkpoint:
                        # Formato alternativo con 'state_dict'
                        state_dict = checkpoint['state_dict']
                        first_key = list(state_dict.keys())[0] if state_dict else ""
                        is_custom_arch = 'enc1' in first_key or 'aspp' in first_key or 'decoder_conv' in first_key
                        
                        if is_custom_arch:
                            print("Reconstruyendo arquitectura DeepLabV3+ personalizada (formato state_dict)...")
                            try:
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ personalizado cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(f"Error cargando modelo DeepLabV3+ personalizado: {e}")
                        else:
                            print("Reconstruyendo arquitectura DeepLabV3+ estándar (formato state_dict)...")
                            try:
                                from torchvision.models.segmentation import deeplabv3_resnet50
                                self.model = deeplabv3_resnet50(
                                    num_classes=NUM_CLASSES,
                                    pretrained_backbone=False
                                )
                                self.model.load_state_dict(state_dict)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print("Modelo DeepLabV3+ ResNet50 cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(f"Error cargando modelo: {e}")
                    # Caso 2: Checkpoint es directamente un state_dict sin wrapper (como DeepLabV3pp)
                    elif first_key.startswith("enc1") or first_key.startswith("conv0_0"):
                        # Es un state_dict directo de una arquitectura personalizada
                        state_dict = checkpoint
                        architecture_name = self.model_config["architecture"]
                        
                        # Detectar arquitectura por las keys
                        is_deeplab = 'enc1' in first_key and ('aspp' in first_key or 'decoder' in first_key)
                        is_unetpp = 'conv0_0' in first_key or 'conv0_1' in first_key
                        
                        print(f"Detectado state_dict directo - Reconstruyendo arquitectura {architecture_name}...")
                        
                        if architecture_name == "DeepLabV3Plus" or is_deeplab:
                            try:
                                self.model = DeepLabV3Plus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                self.model.load_state_dict(state_dict, strict=False)
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                raise RuntimeError(
                                    f"Error cargando modelo DeepLabV3+: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        elif architecture_name == "UNetPlusPlus" or is_unetpp:
                            try:
                                print(f"Construyendo arquitectura UNetPlusPlus...")
                                self.model = UNetPlusPlus(
                                    in_channels=3,
                                    num_classes=NUM_CLASSES
                                )
                                print(f"Cargando state_dict ({len(state_dict)} parámetros)...")
                                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                if missing_keys:
                                    print(f"⚠️  Advertencia: {len(missing_keys)} keys faltantes")
                                if unexpected_keys:
                                    print(f"⚠️  Advertencia: {len(unexpected_keys)} keys inesperadas")
                                
                                self.model = self.model.to(self.device)
                                self.model.eval()
                                print(f"✅ Modelo {self.model_config['name']} cargado exitosamente")
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                raise RuntimeError(
                                    f"Error cargando modelo UNet++: {str(e)}\n"
                                    f"Verifica que la estructura del state_dict coincida con la arquitectura."
                                )
                        else:
                            raise ValueError(f"Arquitectura '{architecture_name}' no soportada para state_dict directo.")
                    else:
                        # Si es un dict pero no tiene las keys esperadas, asumir que es el modelo completo
                        # Esto es poco probable pero lo manejamos
                        raise ValueError(f"Formato de checkpoint no reconocido. Keys encontradas: {list(checkpoint.keys())[:10]}")
                else:
                    # Si no es dict, asumir que es el modelo completo
                    self.model = checkpoint
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                
                # Verificar que el modelo se cargó correctamente
                if not hasattr(self.model, 'forward'):
                    raise RuntimeError("El modelo cargado no tiene método 'forward'")
                
                self.model_loaded = True
                print(f"Modelo cargado exitosamente en {self.device}")
                
            except Exception as e:
                print(f"Error cargando modelo PyTorch: {e}")
                print("Intentando otros formatos...")
                # Aquí podrías agregar lógica para otros formatos (TensorFlow, ONNX, etc.)
                raise
        
        except Exception as e:
            print(f"Error en load_model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa la imagen para el modelo"""
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Ajustar tamaño según el modelo (algunos modelos fueron entrenados con 256x256)
        if self.model_config.get("architecture") == "DeepLabV3PlusDenseDecoder":
            # Este modelo fue entrenado con 256x256 según el JSON de clases
            input_size = (256, 256)
        else:
            input_size = INPUT_SIZE
        
        # Redimensionar al tamaño de entrada del modelo
        image = image.resize(input_size, Image.Resampling.BILINEAR)
        
        # Convertir a numpy array y normalizar
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convertir de HWC a CHW y agregar batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def postprocess_prediction(self, prediction: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocesa la predicción del modelo con mejoras para T1"""
        # Obtener la clase predicha para cada pixel
        if prediction.dim() > 3:
            prediction = prediction.squeeze(0)
        
        if prediction.dim() == 3:
            # Obtener probabilidades
            probs = torch.softmax(prediction, dim=0).cpu().numpy()  # [C, H, W]
            
            # Argmax estándar
            pred_mask = np.argmax(probs, axis=0)
            
            # Mejora especial para T1 si existe - UMBRALES MUY BAJOS para modelos mal entrenados
            t1_class = self._get_class_index("T1")
            if t1_class is not None and t1_class < probs.shape[0]:
                t1_probs = probs[t1_class]
                bg_class = self._get_class_index("F") or self._get_class_index("Background") or 0
                v_class_idx = self._get_class_index("V")
                bg_probs = probs[bg_class] if bg_class < probs.shape[0] else np.zeros_like(t1_probs)
                v_probs = probs[v_class_idx] if v_class_idx is not None and v_class_idx < probs.shape[0] else np.zeros_like(t1_probs)
                max_probs = np.max(probs, axis=0)
                
                # Estrategia 1: T1 con prob > umbral MUY bajo (adaptativo según modelo)
                # Para modelos con probabilidades muy bajas de T1, usar umbrales extremadamente bajos
                t1_relative = t1_probs / (max_probs + 1e-8)
                
                # Detectar si el modelo tiene probabilidades muy bajas de T1
                t1_max_prob = t1_probs.max()
                if t1_max_prob < 0.01:  # Si la prob máxima de T1 es < 1%, usar umbrales muy bajos
                    t1_abs_threshold = 0.0005  # 0.05%
                    t1_rel_threshold = 0.05  # 5% relativo
                    t1_bg_factor = 0.01  # 1% de background
                elif t1_max_prob < 0.05:  # Si la prob máxima es < 5%, usar umbrales bajos
                    t1_abs_threshold = 0.002  # 0.2%
                    t1_rel_threshold = 0.08  # 8% relativo
                    t1_bg_factor = 0.05  # 5% de background
                else:  # Umbrales normales
                    t1_abs_threshold = 0.05  # 5%
                    t1_rel_threshold = 0.15  # 15% relativo
                    t1_bg_factor = 0.3  # 30% de background
                
                t1_condition1 = (
                    (t1_probs > t1_abs_threshold) &  # Umbral absoluto adaptativo
                    (t1_relative > t1_rel_threshold) &  # Relativa adaptativa
                    (t1_probs > bg_probs * t1_bg_factor)  # Mayor que Background * factor
                )
                
                # Estrategia 2: T1 es segunda clase más probable con prob > umbral adaptativo
                sorted_indices = np.argsort(probs, axis=0)[::-1]
                second_class = sorted_indices[1]
                t1_second_threshold = max(0.0003, t1_abs_threshold * 0.6)  # 60% del umbral absoluto
                t1_is_second = (second_class == t1_class) & (t1_probs > t1_second_threshold)
                
                # Estrategia 3: T1 tiene prob > umbral adaptativo
                t1_high_threshold = max(0.0005, t1_abs_threshold * 1.2)  # 20% más que el umbral absoluto
                t1_high_prob = t1_probs > t1_high_threshold
                
                # Estrategia 4: T1 es al menos un porcentaje de la suma de todas las probabilidades
                total_probs = np.sum(probs, axis=0)
                t1_ratio = t1_probs / (total_probs + 1e-8)
                t1_ratio_threshold = max(0.05, t1_rel_threshold * 0.5)  # 50% del umbral relativo
                t1_significant = (t1_ratio > t1_ratio_threshold) & (t1_probs > t1_abs_threshold)
                
                # Combinar todas las estrategias
                t1_mask = t1_condition1 | t1_is_second | t1_high_prob | t1_significant
                pred_mask[t1_mask] = t1_class
            else:
                # Argmax estándar si no hay T1
                pred_mask = np.argmax(probs, axis=0)
        else:
            # Si ya es [H, W]
            pred_mask = prediction.cpu().numpy()
        
        # Redimensionar a tamaño original
        pred_mask = cv2.resize(
            pred_mask.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        return pred_mask
    
    def predict(self, image: Image.Image, return_probs: bool = False):
        """
        Realiza la predicción de segmentación
        
        Args:
            image: Imagen a segmentar
            return_probs: Si True, retorna también las probabilidades
            
        Returns:
            mask: Máscara de segmentación (H, W) o tupla (mask, probs) si return_probs=True
        """
        if not self.model_loaded:
            self.load_model()
        
        original_size = image.size
        
        # Preprocesar
        input_tensor = self.preprocess_image(image)
        
        # Predecir
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Si el output es un dict (formato estándar de torchvision segmentation models)
            if isinstance(output, dict):
                output = output['out']
        
        # Obtener probabilidades antes del argmax
        probs = torch.softmax(output, dim=1) if output.dim() > 2 else None
        
        # Postprocesar
        mask = self.postprocess_prediction(output, original_size)
        
        if return_probs and probs is not None:
            # Redimensionar probabilidades al tamaño original
            probs_np = probs.squeeze(0).cpu().numpy()  # [C, H, W]
            probs_resized = np.zeros((probs_np.shape[0], original_size[1], original_size[0]))
            for c in range(probs_np.shape[0]):
                probs_resized[c] = cv2.resize(
                    probs_np[c],
                    original_size,
                    interpolation=cv2.INTER_LINEAR
                )
            return mask, probs_resized
        
        return mask
    
    def create_visualization(self, original_image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Crea una visualización de la segmentación superpuesta sobre la imagen original"""
        # Convertir imagen original a numpy y asegurar tipo uint8
        img_array = np.array(original_image.convert('RGB')).astype(np.uint8)
        
        # Asegurar que mask sea int y tenga valores válidos
        mask = mask.astype(np.int32)
        mask = np.clip(mask, 0, len(self.classes) - 1)
        
        # Redimensionar mask si es necesario para que coincida con la imagen
        if mask.shape[:2] != img_array.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img_array.shape[1], img_array.shape[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(np.int32)
        
        # Crear colores según la imagen de referencia: V (columna) en verde, T1 en rojo
        # Mapear colores según el nombre de la clase
        colors = np.zeros((len(self.classes), 3), dtype=np.uint8)
        
        for i, class_name in enumerate(self.classes):
            class_name_lower = class_name.lower()
            # Mapear colores según el formato de deeplab_resnet50:
            # 0: "F" (Fondo) -> negro
            # 1: "V" (Columna) -> verde
            # 2: "T1" (Vértebra T1) -> rojo
            if 'background' in class_name_lower or 'fondo' in class_name_lower or class_name == 'F' or class_name == '0':
                colors[i] = [0, 0, 0]  # Background/F - negro
            elif 'v' in class_name_lower or 'columna' in class_name_lower or class_name == 'V' or class_name == '1':
                colors[i] = [0, 255, 0]  # V (Columna) - verde
            elif 't1' in class_name_lower or class_name == 'T1' or class_name == '2':
                colors[i] = [255, 0, 0]  # T1 - rojo
            else:
                # Color por defecto si no coincide
                colors[i] = [128, 128, 128]  # Gris
        
        # Crear imagen de segmentación coloreada
        # Asegurar que mask tenga la forma correcta para indexar
        if mask.ndim == 2:
            colored_mask = colors[mask]
        else:
            # Si mask tiene más dimensiones, tomar solo la primera
            colored_mask = colors[mask.reshape(-1)].reshape(*mask.shape, 3)
        
        # Asegurar que colored_mask sea uint8
        colored_mask = colored_mask.astype(np.uint8)
        
        # Verificar que las dimensiones coincidan
        if colored_mask.shape[:2] != img_array.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (img_array.shape[1], img_array.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
        
        # Superponer con transparencia (ambos deben ser uint8 del mismo tamaño)
        # Asegurar que ambos arrays sean uint8 y tengan las mismas dimensiones
        img_uint8 = img_array.astype(np.uint8)
        mask_uint8 = colored_mask.astype(np.uint8)
        
        # Verificar dimensiones una última vez
        if img_uint8.shape != mask_uint8.shape:
            mask_uint8 = cv2.resize(mask_uint8, (img_uint8.shape[1], img_uint8.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Superponer con transparencia
        overlay = cv2.addWeighted(img_uint8, 0.6, mask_uint8, 0.4, 0)
        
        return Image.fromarray(overlay)
    
    def calculate_metrics(self, mask: np.ndarray, probs: Optional[np.ndarray] = None) -> dict:
        """
        Calcula métricas de la segmentación incluyendo IoU y Dice estimados basados en confianza
        
        Args:
            mask: Máscara de segmentación (H, W) con valores de clase
            probs: Probabilidades del modelo [C, H, W] (opcional)
            
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {}
        
        # Calcular distribución de clases
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        # Porcentaje de cada clase
        class_percentages = {}
        for cls, count in zip(unique_classes, counts):
            if cls < len(self.classes):
                class_name = self.classes[int(cls)]
                percentage = float(count / total_pixels * 100)
                class_percentages[class_name] = percentage
                metrics[f"{class_name}_percentage"] = percentage
                metrics[f"{class_name}_pixels"] = int(count)
        
        # Calcular cobertura total (todo excepto background)
        background_pixels = counts[unique_classes == 0].sum() if 0 in unique_classes else 0
        foreground_pixels = total_pixels - background_pixels
        metrics["foreground_coverage"] = float(foreground_pixels / total_pixels * 100)
        
        # Calcular métricas básicas
        metrics["total_classes_detected"] = len(unique_classes)
        metrics["total_pixels"] = int(total_pixels)
        
        # Calcular IoU y Dice basados en probabilidades del modelo
        # NOTA: Sin ground truth, usamos las probabilidades del modelo como referencia
        # Esto da una estimación de qué tan "seguro" está el modelo de sus predicciones
        if probs is not None and probs.shape[0] == len(self.classes):
            # probs es [C, H, W]
            # Calcular métricas por clase
            for c in range(len(self.classes)):
                if c < probs.shape[0]:
                    class_name = self.classes[c]
                    # Máscara binaria predicha para esta clase
                    pred_mask = (mask == c).astype(np.float32)
                    
                    # Probabilidades de esta clase [H, W]
                    class_probs = probs[c]
                    
                    # Calcular confianza promedio en píxeles predichos
                    if pred_mask.sum() > 0:
                        avg_confidence = float(class_probs[pred_mask.astype(bool)].mean())
                        metrics[f"{class_name}_confidence"] = avg_confidence
                    else:
                        avg_confidence = 0.0
                        metrics[f"{class_name}_confidence"] = 0.0
                    
                    # Calcular IoU y Dice basados en probabilidades (similar al notebook)
                    # NOTA: Sin ground truth, usamos las probabilidades como referencia
                    # El método compara la predicción con regiones de alta probabilidad
                    
                    # Método principal: IoU basado en intersección/unión con probabilidades
                    # Similar al notebook: inter = (pred & high_prob), union = (pred | high_prob)
                    # Usar umbral adaptativo: más bajo para clases minoritarias (V, T1)
                    is_minority_class = class_name in ['V', 'T1']
                    prob_threshold = 0.4 if is_minority_class else 0.5
                    
                    # Máscara de alta probabilidad
                    prob_mask = (class_probs > prob_threshold).astype(np.float32)
                    
                    # Calcular intersección y unión (similar al notebook)
                    # Intersección: píxeles predichos Y con probabilidad alta
                    intersection = (pred_mask * prob_mask).sum()
                    # Unión: píxeles predichos O con probabilidad alta
                    union = np.maximum(pred_mask, prob_mask).sum()
                    
                    # IoU: intersection / union (igual que en el notebook)
                    if union > 0:
                        iou = float(intersection / (union + 1e-7))
                    else:
                        iou = 0.0
                    
                    metrics[f"{class_name}_iou"] = iou
                    metrics[f"{class_name}_iou_estimated"] = iou  # Mantener compatibilidad
                    
                    # Calcular Dice: 2 * intersection / (pred_sum + prob_sum)
                    # Similar al notebook pero usando probabilidades en lugar de ground truth
                    pred_sum = pred_mask.sum()
                    prob_sum = prob_mask.sum()
                    if (pred_sum + prob_sum) > 0:
                        dice = float((2.0 * intersection) / (pred_sum + prob_sum + 1e-7))
                    else:
                        dice = 0.0
                    
                    metrics[f"{class_name}_dice"] = dice
                    metrics[f"{class_name}_dice_estimated"] = dice  # Mantener compatibilidad
            
            # Calcular IoU y Dice promedio (sin background/F)
            bg_class = self._get_class_index("F") or self._get_class_index("Background") or 0
            foreground_classes = [c for c in range(len(self.classes)) if c != bg_class]
            
            ious = [metrics.get(f"{self.classes[c]}_iou", 0.0) for c in foreground_classes]
            dices = [metrics.get(f"{self.classes[c]}_dice", 0.0) for c in foreground_classes]
            
            if ious:
                metrics["mean_iou"] = float(np.mean(ious))
                metrics["mean_iou_estimated"] = float(np.mean(ious))  # Mantener compatibilidad
            if dices:
                metrics["mean_dice"] = float(np.mean(dices))
                metrics["mean_dice_estimated"] = float(np.mean(dices))  # Mantener compatibilidad
        
        # Calcular entropía de la distribución
        if len(unique_classes) > 1:
            non_bg_classes = [c for c in unique_classes if c != 0]
            if len(non_bg_classes) > 0:
                class_probs = [counts[unique_classes == c][0] / total_pixels for c in non_bg_classes]
                entropy = -sum(p * np.log2(p + 1e-10) for p in class_probs)
                metrics["prediction_entropy"] = float(entropy)
        
        return metrics
    
    def _get_class_index(self, class_name: str) -> Optional[int]:
        """Obtiene el índice de una clase por su nombre"""
        for i, name in enumerate(self.classes):
            if class_name.lower() in name.lower() or name.lower() in class_name.lower():
                return i
        return None
    
    def improve_v_segmentation(self, mask: np.ndarray, probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mejora la segmentación de V (columna) limpiando ruido y conectando regiones
        
        Args:
            mask: Máscara de segmentación original
            probs: Probabilidades del modelo [C, H, W] (opcional)
            
        Returns:
            Máscara mejorada
        """
        improved_mask = mask.copy()
        
        # Buscar dinámicamente el índice de V
        v_class = self._get_class_index("V")
        if v_class is None:
            return improved_mask
        
        # Crear máscara binaria de V
        v_mask = (improved_mask == v_class).astype(np.uint8)
        
        if v_mask.sum() == 0:
            return improved_mask
        
        # 1. Eliminar ruido pequeño (opening)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        v_cleaned = cv2.morphologyEx(v_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
        
        # 2. Rellenar huecos pequeños (closing)
        v_filled = cv2.morphologyEx(v_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # 3. Si tenemos probabilidades, filtrar por confianza
        if probs is not None and v_class < probs.shape[0]:
            v_probs = probs[v_class]
            # Mantener solo píxeles con confianza > 0.5
            high_confidence = (v_probs > 0.5).astype(np.uint8)
            v_filled = np.logical_and(v_filled, high_confidence).astype(np.uint8)
        
        # 4. Filtrar componentes muy pequeños (< 500 píxeles)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(v_filled, connectivity=8)
        v_filtered = np.zeros_like(v_filled)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area >= 500:  # Mantener solo componentes grandes
                v_filtered[labels == label_id] = 1
        
        # 5. Conectar regiones cercanas de la columna (dilatación controlada)
        kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        v_connected = cv2.dilate(v_filtered, kernel_connect, iterations=1)
        v_connected = cv2.erode(v_connected, kernel_connect, iterations=1)
        
        # 6. Si tenemos probabilidades, filtrar la expansión por confianza
        if probs is not None and v_class < probs.shape[0]:
            v_probs = probs[v_class]
            v_connected = v_connected & (v_probs > 0.4).astype(np.uint8)
        
        # Actualizar máscara
        improved_mask[v_connected == 1] = v_class
        
        return improved_mask
    
    def improve_t1_segmentation(self, mask: np.ndarray, probs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Mejora la segmentación de T1 usando post-procesamiento agresivo
        
        Args:
            mask: Máscara de segmentación original
            probs: Probabilidades del modelo [C, H, W] (opcional)
            
        Returns:
            Máscara mejorada
        """
        improved_mask = mask.copy()
        
        # Buscar dinámicamente el índice de T1
        t1_class = self._get_class_index("T1")
        if t1_class is None:
            return improved_mask
        
        # Si no tenemos probabilidades, usar solo operaciones morfológicas
        if probs is None or t1_class >= probs.shape[0]:
            # Crear máscara binaria de T1
            t1_mask = (improved_mask == t1_class).astype(np.uint8)
            
            # Operaciones morfológicas para mejorar T1
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            t1_cleaned = cv2.morphologyEx(t1_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
            t1_filled = cv2.morphologyEx(t1_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=1)
            
            improved_mask[t1_filled == 1] = t1_class
            return improved_mask
        
        # Si tenemos probabilidades, usar estrategia más agresiva
        t1_probs = probs[t1_class]
        bg_class = self._get_class_index("F") or self._get_class_index("Background") or 0
        v_class_idx = self._get_class_index("V")
        bg_probs = probs[bg_class] if bg_class < probs.shape[0] else np.zeros_like(t1_probs)
        v_probs = probs[v_class_idx] if v_class_idx is not None and v_class_idx < probs.shape[0] else np.zeros_like(t1_probs)
        
        # Estrategia 1: Detectar T1 usando umbrales adaptativos
        # Ajustar umbrales según la probabilidad máxima de T1
        max_probs = np.max(probs, axis=0)
        t1_relative = t1_probs / (max_probs + 1e-8)
        
        # Detectar si el modelo tiene probabilidades muy bajas de T1
        t1_max_prob = t1_probs.max()
        if t1_max_prob < 0.01:  # Si la prob máxima de T1 es < 1%, usar umbrales muy bajos
            t1_abs_threshold = 0.0005  # 0.05%
            t1_rel_threshold = 0.05  # 5% relativo
            t1_bg_factor = 0.01  # 1% de background
            t1_v_factor = 0.1  # 10% de V
        elif t1_max_prob < 0.05:  # Si la prob máxima es < 5%, usar umbrales bajos
            t1_abs_threshold = 0.002  # 0.2%
            t1_rel_threshold = 0.08  # 8% relativo
            t1_bg_factor = 0.05  # 5% de background
            t1_v_factor = 0.3  # 30% de V
        else:  # Umbrales normales
            t1_abs_threshold = 0.07  # 7%
            t1_rel_threshold = 0.25  # 25% relativo
            t1_bg_factor = 0.5  # 50% de background
            t1_v_factor = 0.6  # 60% de V
        
        t1_candidates = (
            (t1_probs > t1_abs_threshold) &  # Umbral absoluto adaptativo
            (t1_relative > t1_rel_threshold) &  # Relativo adaptativo
            (t1_probs > bg_probs * t1_bg_factor)  # Mayor que Background * factor
        )
        
        # Estrategia 2: Si T1 es la segunda clase más probable y tiene prob > umbral adaptativo
        sorted_indices = np.argsort(probs, axis=0)[::-1]
        second_class = sorted_indices[1]
        t1_second_threshold = max(0.0003, t1_abs_threshold * 0.6)  # 60% del umbral absoluto
        t1_is_second = (second_class == t1_class) & (t1_probs > t1_second_threshold)
        
        # Estrategia 3: Regiones donde T1 tiene prob > umbral adaptativo Y es mayor que V
        t1_high_threshold = max(0.0005, t1_abs_threshold * 1.2)  # 20% más que el umbral absoluto
        t1_high_prob = (t1_probs > t1_high_threshold) & (t1_probs > v_probs * t1_v_factor)
        
        # Estrategia 4: T1 es al menos un porcentaje de la suma total de probabilidades
        total_probs = np.sum(probs, axis=0)
        t1_ratio = t1_probs / (total_probs + 1e-8)
        t1_ratio_threshold = max(0.05, t1_rel_threshold * 0.5)  # 50% del umbral relativo
        t1_significant = (t1_ratio > t1_ratio_threshold) & (t1_probs > t1_abs_threshold)
        
        # Estrategia 5: T1 cerca de regiones de V (T1 está arriba de la columna) - más restrictivo
        # Si hay V detectado, buscar T1 en la parte superior de la imagen
        v_class_idx = self._get_class_index("V")
        if v_class_idx is not None:
            v_mask = (mask == v_class_idx).astype(np.uint8)
        else:
            v_mask = np.zeros_like(mask, dtype=np.uint8)
        if v_mask.sum() > 0:
            # Encontrar la parte superior de V
            v_coords = np.where(v_mask > 0)
            if len(v_coords[0]) > 0:
                v_top = v_coords[0].min()
                # Buscar T1 en la región superior (arriba de V) con umbral adaptativo
                top_region = np.zeros_like(t1_probs, dtype=bool)
                top_region[:max(v_top + 20, mask.shape[0] // 6), :] = True
                # T1 debe tener prob > umbral adaptativo en esta región y ser mayor que V
                t1_near_v_threshold = max(0.0005, t1_abs_threshold * 1.2)  # 20% más que el umbral absoluto
                t1_near_v = (t1_probs > t1_near_v_threshold) & top_region & (t1_probs > v_probs * t1_v_factor)
                t1_significant = t1_significant | t1_near_v
        
        # Combinar todas las estrategias
        t1_final_mask = t1_candidates | t1_is_second | t1_high_prob | t1_significant
        
        # Operaciones morfológicas para limpiar y conectar regiones
        t1_binary = t1_final_mask.astype(np.uint8)
        
        # 1. Eliminar ruido muy pequeño (opening con kernel pequeño)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        t1_cleaned = cv2.morphologyEx(t1_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 2. Rellenar huecos pequeños (closing)
        t1_filled = cv2.morphologyEx(t1_cleaned, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # 3. Conectar regiones cercanas (dilation seguido de erosion)
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        t1_connected = cv2.dilate(t1_filled, kernel_medium, iterations=1)
        t1_connected = cv2.erode(t1_connected, kernel_medium, iterations=1)
        
        # 4. Filtrar componentes muy pequeños y mantener solo los más grandes
        # T1 debería ser una región compacta, no fragmentada
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(t1_connected, connectivity=8)
        t1_filtered = np.zeros_like(t1_connected)
        
        if num_labels > 1:
            # Ordenar componentes por área
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_indices = np.argsort(areas)[::-1]  # De mayor a menor
            
            # Mantener solo los componentes más grandes (top 3 o los que tengan > 200 píxeles)
            for idx in sorted_indices[:3]:  # Top 3 componentes
                label_id = idx + 1  # +1 porque saltamos label 0
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area >= 200:  # Componentes con al menos 200 píxeles
                    t1_filtered[labels == label_id] = 1
            # Si no hay componentes grandes, mantener el más grande si tiene > 100 píxeles
            if t1_filtered.sum() == 0 and len(sorted_indices) > 0:
                largest_idx = sorted_indices[0] + 1
                if stats[largest_idx, cv2.CC_STAT_AREA] >= 100:
                    t1_filtered[labels == largest_idx] = 1
        
        # Actualizar máscara mejorada
        improved_mask[t1_filtered == 1] = t1_class
        
        # Mantener T1 que ya estaba en la máscara original si tiene probabilidad razonable
        existing_t1 = (mask == t1_class).astype(np.uint8)
        if existing_t1.sum() > 0:
            # Si T1 existente tiene prob > 0.06, mantenerlo
            existing_t1_keep = existing_t1 & (t1_probs > 0.06).astype(np.uint8)
            improved_mask[existing_t1_keep == 1] = t1_class
        
        # Estrategia adicional: Expandir T1 detectado usando dilation (muy controlado)
        if t1_filtered.sum() > 0:
            # Dilatar ligeramente para capturar T1 completo
            kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            t1_expanded = cv2.dilate(t1_filtered, kernel_expand, iterations=1)
            # Pero solo mantener donde T1 tiene prob > 0.07 (umbral más alto)
            t1_expanded_filtered = t1_expanded & (t1_probs > 0.07).astype(np.uint8)
            improved_mask[t1_expanded_filtered == 1] = t1_class
        
        # Estrategia final: Si hay V detectado, buscar T1 en la parte superior (muy restrictivo)
        v_class_idx = self._get_class_index("V")
        if v_class_idx is not None:
            v_mask = (improved_mask == v_class_idx).astype(np.uint8)
        else:
            v_mask = np.zeros_like(improved_mask, dtype=np.uint8)
        if v_mask.sum() > 0:
            v_coords = np.where(v_mask > 0)
            if len(v_coords[0]) > 0:
                v_top = v_coords[0].min()
                # Región superior pequeña donde debería estar T1
                top_region_mask = np.zeros_like(improved_mask, dtype=bool)
                top_region_mask[:max(v_top + 30, mask.shape[0] // 6), :] = True
                
                # Buscar T1 con prob > 0.08 en esta región (umbral alto) y mayor que V
                t1_in_top = (t1_probs > 0.08) & top_region_mask & (t1_probs > v_probs * 0.5)
                improved_mask[t1_in_top] = t1_class
        
        return improved_mask

