import os
import io
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageEnhance
from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from scipy import ndimage
import gc
import cv2

# Stable Diffusion imports
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    AutoPipelineForInpainting,
    DPMSolverMultistepScheduler,
)
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

app = Flask(__name__)

# ================== CONFIG ==================

UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'
MODEL_PATH = 'u2net.pth'

# Model IDs
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
SD_INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
CONTROLNET_TILE_ID = "lllyasviel/control_v11f1e_sd15_tile"
CONTROLNET_INPAINT_ID = "lllyasviel/control_v11p_sd15_inpaint"
IP_ADAPTER_MODEL_ID = "h94/IP-Adapter"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ================== STYLE PRESETS ==================

STYLE_PRESETS = {
    "auto": {
        "prompt": "same style as reference, matching lighting and colors, seamless blend, professional quality",
        "negative": "different style, mismatched colors, obvious edit, artifact, blurry",
        "denoising": 0.5,
        "ip_adapter_scale": 0.6,
        "controlnet_scale": 0.8,
    },
    # NEW: Avatar-style blue cinematic preset
    "avatar_blue": {
        "prompt": (
            "cinematic portrait in a dark movie theater, character lit by strong blue cinematic light, "
            "blue-tinted skin and clothes, monochrome blue color grading, highly detailed, dramatic "
            "lighting, professional photography, Avatar style"
        ),
        "negative": (
            "orange lighting, warm skin tones, normal skin color, yellow, green, brown, washed out, "
            "flat lighting, low contrast, artifacts, cartoon, anime, deformed face, extra limbs"
        ),
        "denoising": 0.55,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.85,
    },
    "avatar_3d": {
        "prompt": "3D rendered avatar, stylized character, smooth skin, matching environment lighting, CGI quality, Pixar style",
        "negative": "realistic photo, 2d, flat, ugly, deformed, noisy",
        "denoising": 0.6,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "realistic": {
        "prompt": "ultra realistic photo, professional photography, natural skin, matching scene lighting, seamless composite",
        "negative": "cartoon, anime, cgi, artificial, fake looking, obvious edit",
        "denoising": 0.45,
        "ip_adapter_scale": 0.5,
        "controlnet_scale": 0.85,
    },
    "artistic": {
        "prompt": "artistic portrait, fine art, painterly style, matching artistic environment, museum quality",
        "negative": "photo realistic, plain, boring, low quality",
        "denoising": 0.55,
        "ip_adapter_scale": 0.65,
        "controlnet_scale": 0.75,
    },
    "fantasy": {
        "prompt": "fantasy character, magical atmosphere, ethereal glow, matching mystical environment, enchanted",
        "negative": "mundane, realistic, boring, low quality",
        "denoising": 0.65,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "anime": {
        "prompt": "anime style, japanese animation, cel shaded, vibrant colors, matching anime background",
        "negative": "realistic, photo, 3d render, western cartoon",
        "denoising": 0.7,
        "ip_adapter_scale": 0.75,
        "controlnet_scale": 0.65,
    },
    "cinematic": {
        "prompt": "cinematic shot, movie quality, dramatic lighting, color graded, matching scene atmosphere, film grain",
        "negative": "amateur, flat lighting, oversaturated, low budget",
        "denoising": 0.5,
        "ip_adapter_scale": 0.6,
        "controlnet_scale": 0.8,
    },
    "cyberpunk": {
        "prompt": "cyberpunk style, neon lighting, futuristic, holographic effects, matching neon environment",
        "negative": "natural, organic, vintage, old fashioned",
        "denoising": 0.6,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "vintage": {
        "prompt": "vintage photo, retro film look, nostalgic colors, film grain, matching period setting",
        "negative": "modern, digital, clean, oversaturated",
        "denoising": 0.5,
        "ip_adapter_scale": 0.6,
        "controlnet_scale": 0.8,
    },
}


# ================== U2-NET ARCHITECTURE ==================

class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx6 = self.rebnconv6(hx5)
        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx5 = self.rebnconv5(hx4)
        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)
        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))
        return hx1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6*out_ch, out_ch, 1)

    def forward(self, x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)
        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)
        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)
        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)
        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)


# ================== GLOBAL MODELS ==================

u2net_model = None
sd_pipeline = None
controlnet_tile = None
ip_adapter_loaded = False

face_cascade = None
eye_cascade = None


# ================== MODEL LOADING ==================

def load_opencv_cascades():
    global face_cascade, eye_cascade
    if face_cascade is None:
        cv2_data_path = cv2.data.haarcascades
        face_cascade = cv2.CascadeClassifier(cv2_data_path + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2_data_path + 'haarcascade_eye.xml')
        print("✅ OpenCV cascades loaded")
    return face_cascade, eye_cascade


def load_u2net_model():
    global u2net_model
    if u2net_model is None:
        print(f"Loading U2-Net from {MODEL_PATH}...")
        u2net_model = U2NET(3, 1)
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            u2net_model.load_state_dict(state_dict)
            print("✅ U2-Net loaded!")
        else:
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        u2net_model.to(DEVICE)
        u2net_model.eval()
    return u2net_model


def load_sd_pipeline_with_ip_adapter():
    """
    Load Stable Diffusion Inpainting with ControlNet Tile and IP-Adapter
    This is the key for style transfer from template to person
    """
    global sd_pipeline, controlnet_tile, ip_adapter_loaded
    
    if sd_pipeline is None:
        print("Loading SD Inpainting + ControlNet Tile + IP-Adapter...")
        
        # Load ControlNet Tile for detail preservation
        print("  Loading ControlNet Tile...")
        controlnet_tile = ControlNetModel.from_pretrained(
            CONTROLNET_TILE_ID,
            torch_dtype=DTYPE
        )
        
        # Load SD Inpainting with ControlNet
        print("  Loading SD Inpainting Pipeline...")
        sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL_ID,
            controlnet=controlnet_tile,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use faster scheduler
        sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipeline.scheduler.config
        )
        
        sd_pipeline = sd_pipeline.to(DEVICE)
        
        # Load IP-Adapter for style transfer
        print("  Loading IP-Adapter...")
        sd_pipeline.load_ip_adapter(
            IP_ADAPTER_MODEL_ID,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        ip_adapter_loaded = True
        
        # Memory optimizations
        if torch.cuda.is_available():
            sd_pipeline.enable_attention_slicing()
            try:
                sd_pipeline.enable_xformers_memory_efficient_attention()
                print("  ✅ xformers enabled")
            except Exception:
                pass
        
        print("✅ Full pipeline loaded: SD Inpaint + ControlNet Tile + IP-Adapter")
    
    return sd_pipeline


def load_sd_pipeline_simple():
    """Fallback: Load simple SD Inpainting without ControlNet/IP-Adapter"""
    global sd_pipeline
    
    if sd_pipeline is None:
        print("Loading simple SD Inpainting pipeline...")
        
        sd_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL_ID,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipeline.scheduler.config
        )
        
        sd_pipeline = sd_pipeline.to(DEVICE)
        
        if torch.cuda.is_available():
            sd_pipeline.enable_attention_slicing()
        
        print("✅ Simple SD Inpainting loaded")
    
    return sd_pipeline


def unload_pipelines():
    global sd_pipeline, controlnet_tile, ip_adapter_loaded
    if sd_pipeline is not None:
        del sd_pipeline
        sd_pipeline = None
    if controlnet_tile is not None:
        del controlnet_tile
        controlnet_tile = None
    ip_adapter_loaded = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🗑️ Pipelines unloaded")


# ================== IMAGE PROCESSING ==================

def create_tile_condition(image, resolution=512):
    """Create tile condition image for ControlNet Tile"""
    img = image.convert('RGB')
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def detect_faces_and_eyes(image):
    """Detect faces and eyes for protection mask"""
    load_opencv_cascades()
    
    img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    all_eyes = []
    all_faces = []
    
    for (fx, fy, fw, fh) in faces:
        all_faces.append((fx, fy, fw, fh))
        roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        
        for (ex, ey, ew, eh) in eyes:
            padding = int(max(ew, eh) * 0.5)
            all_eyes.append((
                max(0, fx + ex - padding),
                max(0, fy + ey - padding),
                ew + 2 * padding,
                eh + 2 * padding
            ))
    
    return all_faces, all_eyes


def create_eye_protection_mask(image_size, eye_regions, blur_radius=10):
    """Create mask to protect eyes (white = protect, keep original)"""
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for (x, y, w, h) in eye_regions:
        draw.ellipse([x, y, x + w, y + h], fill=255)
    
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask


def get_edge_mask(alpha, edge_width=5):
    dilated = ndimage.binary_dilation(alpha > 127, iterations=edge_width)
    eroded = ndimage.binary_erosion(alpha > 127, iterations=edge_width)
    return dilated & ~eroded


def refine_mask(mask, erode_size=3, blur_radius=2, threshold=180):
    mask_np = np.array(mask).astype(np.float32)
    mask_np = np.where(mask_np > threshold, 255, 0).astype(np.uint8)
    
    struct = ndimage.generate_binary_structure(2, 1)
    mask_bool = mask_np > 127
    
    if erode_size > 0:
        mask_bool = ndimage.binary_erosion(mask_bool, struct, iterations=erode_size)
    
    mask_np = (mask_bool * 255).astype(np.uint8)
    mask_refined = Image.fromarray(mask_np)
    
    if blur_radius > 0:
        mask_refined = mask_refined.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask_refined


def decontaminate_edges(image_rgba, edge_width=5, iterations=3):
    arr = np.array(image_rgba).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    
    edge_mask = get_edge_mask(alpha, edge_width)
    result_rgb = rgb.copy()
    
    for _ in range(iterations):
        for c in range(3):
            blurred = ndimage.uniform_filter(result_rgb[:, :, c] * (alpha > 50), size=edge_width * 2)
            count = ndimage.uniform_filter((alpha > 50).astype(float), size=edge_width * 2)
            count = np.maximum(count, 0.001)
            blurred /= count
            result_rgb[:, :, c] = np.where(edge_mask, blurred, result_rgb[:, :, c])
    
    arr[:, :, :3] = np.clip(result_rgb, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# ================== U2NET BACKGROUND REMOVAL ==================

def preprocess_for_u2net(image, target_size=320):
    w, h = image.size
    if w > h:
        new_w, new_h = target_size, int(h * target_size / w)
    else:
        new_h, new_w = target_size, int(w * target_size / h)
    
    image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(image_resized, (paste_x, paste_y))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(canvas).unsqueeze(0), (paste_x, paste_y, new_w, new_h, target_size)


def generate_u2net_mask(image):
    model = load_u2net_model()
    original_size = image.size
    
    input_tensor, crop_info = preprocess_for_u2net(image)
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        d0, *_ = model(input_tensor)
    
    mask = d0.squeeze().cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = (mask * 255).astype(np.uint8)
    
    mask_pil = Image.fromarray(mask)
    paste_x, paste_y, new_w, new_h, target_size = crop_info
    mask_cropped = mask_pil.crop((paste_x, paste_y, paste_x + new_w, paste_y + new_h))
    mask_final = mask_cropped.resize(original_size, Image.Resampling.LANCZOS)
    
    return mask_final


def remove_background(image, erode_size=3, feather=2):
    """Step 1: Remove background using U2Net"""
    mask = generate_u2net_mask(image)
    mask_refined = refine_mask(mask, erode_size=erode_size, blur_radius=feather)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    mask_feathered = mask_refined.filter(ImageFilter.GaussianBlur(radius=feather))
    rgba = image.copy()
    rgba.putalpha(mask_feathered)
    
    rgba = decontaminate_edges(rgba, edge_width=3, iterations=2)
    
    return rgba, mask_refined


# ================== COMPOSITION ==================

def place_person_on_template(person_rgba, template_path=None, position='center', scale=0.85, padding=0.05):
    """Step 2: Place person on template"""
    
    # Load template
    if template_path and os.path.exists(template_path):
        template = Image.open(template_path).convert('RGBA')
    else:
        # Create default gradient template
        template = Image.new('RGBA', (1080, 1920), (100, 150, 200, 255))
        template_np = np.array(template)
        for y in range(template_np.shape[0]):
            ratio = y / template_np.shape[0]
            template_np[y, :, 0] = int(80 + ratio * 40)
            template_np[y, :, 1] = int(120 + ratio * 60)
            template_np[y, :, 2] = int(180 + ratio * 40)
        template = Image.fromarray(template_np)
    
    template_width, template_height = template.size
    original_template = template.copy()
    
    # Crop person to bounding box
    person_arr = np.array(person_rgba)
    alpha = person_arr[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    
    if not rows.any() or not cols.any():
        return template.convert('RGB'), original_template, None, None, None, None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    person_cropped = person_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    # Scale person
    target_height = int(template_height * scale)
    aspect = person_cropped.width / person_cropped.height
    target_width = int(target_height * aspect)
    
    max_width = int(template_width * (1 - 2 * padding))
    if target_width > max_width:
        target_width = max_width
        target_height = int(target_width / aspect)
    
    person_resized = person_cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Calculate position
    pad_px = int(template_height * padding)
    positions = {
        'center': ((template_width - target_width) // 2, (template_height - target_height) // 2),
        'top-center': ((template_width - target_width) // 2, pad_px),
        'bottom-center': ((template_width - target_width) // 2, template_height - target_height - pad_px),
        'left': (pad_px, template_height - target_height - pad_px),
        'right': (template_width - target_width - pad_px, template_height - target_height - pad_px),
    }
    x, y = positions.get(position, positions['center'])
    
    # Composite
    composite = template.copy()
    composite.paste(person_resized, (x, y), person_resized)
    
    # Create person mask (full size)
    person_mask = Image.new('L', (template_width, template_height), 0)
    person_alpha = person_resized.split()[3]
    person_mask.paste(person_alpha, (x, y))
    person_mask = person_mask.filter(ImageFilter.GaussianBlur(radius=3))
    
    return (
        composite.convert('RGB'),
        original_template,
        (x, y),
        (target_width, target_height),
        person_resized,
        person_mask
    )


# ============ BLUE COLOR GRADE FOR AVATAR LOOK ============

def apply_blue_tone_to_person(composite_image, person_mask, strength=0.9):
    """
    Shift the colors of the person region towards cinematic blue while
    preserving shading. Uses HSV hue shift + saturation/brightness boost.
    Only affects pixels where person_mask > 0.
    """
    if person_mask is None:
        return composite_image

    composite_image = composite_image.convert("RGB")
    w, h = composite_image.size

    # Ensure mask matches composite size
    mask = person_mask.resize((w, h), Image.Resampling.LANCZOS).convert("L")
    mask_np = np.array(mask).astype(np.float32) / 255.0
    if mask_np.max() < 0.05:
        # No person area
        return composite_image

    img_np = np.array(composite_image)
    mask_3d = np.stack([mask_np] * 3, axis=2)

    # Convert to HSV via OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h, s, v = cv2.split(hsv)

    # OpenCV hue range: 0-179, blue ~ 110
    target_hue = 110.0
    h_blue = (1.0 - strength) * h + strength * target_hue
    s_blue = np.clip(s + strength * 70.0, 0, 255)
    v_blue = np.clip(v + strength * 10.0, 0, 255)  # slight brightness lift

    hsv_blue = cv2.merge([h_blue, s_blue, v_blue]).astype(np.uint8)
    img_bgr_blue = cv2.cvtColor(hsv_blue, cv2.COLOR_HSV2BGR)
    img_rgb_blue = cv2.cvtColor(img_bgr_blue, cv2.COLOR_BGR2RGB).astype(np.float32)

    img_orig = img_np.astype(np.float32)
    out = img_orig * (1.0 - mask_3d) + img_rgb_blue * mask_3d
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


# ================== STYLE MATCHING WITH SD + IP-ADAPTER ==================

def style_match_inpaint(
    composite_image,
    original_template,
    person_mask,
    style_preset="auto",
    custom_prompt=None,
    custom_negative=None,
    denoising_strength=None,
    ip_adapter_scale=None,
    controlnet_scale=None,
    num_inference_steps=30,
    preserve_eyes=True,
    eye_regions=None,
    seed=None,
    use_ip_adapter=True
):
    """
    Use SD Inpainting + IP-Adapter + ControlNet Tile to match person to template style.
    """
    
    # Get preset
    preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["auto"])
    
    # Use custom or preset values
    denoising = denoising_strength if denoising_strength is not None else preset["denoising"]
    ip_scale = ip_adapter_scale if ip_adapter_scale is not None else preset["ip_adapter_scale"]
    cn_scale = controlnet_scale if controlnet_scale is not None else preset["controlnet_scale"]
    
    prompt = custom_prompt if custom_prompt else preset["prompt"]
    negative_prompt = custom_negative if custom_negative else preset["negative"]
    
    print(f"🎨 Style: {style_preset}")
    print(f"📝 Prompt: {prompt[:80]}...")
    print(f"⚙️ Denoising: {denoising}, IP-Adapter: {ip_scale}, ControlNet: {cn_scale}")
    
    # Prepare images
    composite_rgb = composite_image.convert('RGB')
    original_size = composite_rgb.size
    
    # Resize for SD (must be multiple of 8)
    w, h = original_size
    max_size = 768
    scale = min(max_size / max(w, h), 1.0)
    new_w = max(64, (int(w * scale) // 8) * 8)
    new_h = max(64, (int(h * scale) // 8) * 8)
    
    composite_resized = composite_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = person_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
    template_resized = original_template.convert('RGB').resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Generator for reproducibility
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
    
    # Try to use full pipeline with IP-Adapter
    if use_ip_adapter:
        try:
            pipeline = load_sd_pipeline_with_ip_adapter()
            
            # Set IP-Adapter scale
            pipeline.set_ip_adapter_scale(ip_scale)
            
            # Create tile condition (the composite itself for structure)
            tile_condition = create_tile_condition(composite_resized, resolution=new_w)
            
            # Run inpainting with IP-Adapter (template as style reference)
            with torch.inference_mode():
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=composite_resized,
                    mask_image=mask_resized,
                    control_image=tile_condition,
                    ip_adapter_image=template_resized,
                    strength=denoising,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=cn_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]
            
            print("✅ Generated with IP-Adapter + ControlNet Tile")
            
        except Exception as e:
            print(f"⚠️ IP-Adapter failed: {e}")
            print("Falling back to simple inpainting...")
            use_ip_adapter = False
    
    if not use_ip_adapter:
        # Fallback to simple inpainting
        pipeline = load_sd_pipeline_simple()
        
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=composite_resized,
                mask_image=mask_resized,
                strength=denoising,
                guidance_scale=7.5,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        
        print("✅ Generated with simple inpainting")
    
    # Resize back
    result = result.resize(original_size, Image.Resampling.LANCZOS)
    
    # === RESTORE ORIGINAL TEMPLATE (background must stay exactly same) ===
    result_np = np.array(result).astype(np.float32)
    template_np = np.array(original_template.convert('RGB').resize(original_size)).astype(np.float32)
    composite_np = np.array(composite_image.convert('RGB')).astype(np.float32)
    
    # Person mask
    mask_np = np.array(person_mask.convert('L')).astype(np.float32) / 255.0
    mask_3d = np.stack([mask_np] * 3, axis=2)
    
    # Replace background with original template
    result_np = result_np * mask_3d + template_np * (1 - mask_3d)
    
    # === PRESERVE EYES ===
    if preserve_eyes and eye_regions:
        eye_mask = create_eye_protection_mask(original_size, eye_regions, blur_radius=8)
        eye_mask_np = np.array(eye_mask).astype(np.float32) / 255.0
        eye_3d = np.stack([eye_mask_np] * 3, axis=2)
        
        # Blend original eyes back (from composite_image)
        result_np = result_np * (1 - eye_3d) + composite_np * eye_3d
    
    result = Image.fromarray(result_np.astype(np.uint8))
    
    return result


# ================== COMPLETE PIPELINE ==================

def process_artgen(
    person_image,
    template_path=None,
    position='center',
    scale=0.85,
    style_preset="auto",
    custom_prompt=None,
    custom_negative=None,
    denoising_strength=None,
    ip_adapter_scale=None,
    controlnet_scale=None,
    num_steps=30,
    preserve_eyes=True,
    use_ip_adapter=True,
    erode_size=3,
    feather=2,
    seed=None
):
    """
    Complete Art-Gen Pipeline:
    
    1. Remove background with U2Net
    2. Place person on template
    3. (Optional) Apply blue color grade to person (for avatar_blue style)
    4. Detect eyes (for protection)
    5. Inpaint with IP-Adapter (template as style reference) + ControlNet Tile
    """
    
    print("=" * 50)
    print("🎨 ART-GEN PIPELINE")
    print("=" * 50)
    
    # STEP 1: Background Removal
    print("\n📸 Step 1: Removing background with U2Net...")
    person_rgba, _ = remove_background(person_image, erode_size=erode_size, feather=feather)
    print("   ✓ Background removed")
    
    # STEP 2: Composition
    print("\n🖼️ Step 2: Placing person on template...")
    composite, original_template, fg_pos, fg_size, person_resized, person_mask = place_person_on_template(
        person_rgba,
        template_path=template_path,
        position=position,
        scale=scale
    )
    
    if fg_pos is None:
        print("   ⚠️ No person detected, returning composite")
        return composite
    
    print(f"   ✓ Person placed at {fg_pos}, size {fg_size}")
    
    # STEP 3: Apply blue color grade if avatar_blue style
    if style_preset == "avatar_blue":
        print("\n🔵 Step 3: Applying blue cinematic color grade to person...")
        composite = apply_blue_tone_to_person(composite, person_mask, strength=0.9)
        print("   ✓ Blue tone applied to person region")
    
    # STEP 4: Detect eyes for preservation
    eye_regions = []
    if preserve_eyes and person_resized:
        print("\n👁️ Step 4: Detecting eyes...")
        _, detected_eyes = detect_faces_and_eyes(person_resized.convert('RGB'))
        px, py = fg_pos
        for (ex, ey, ew, eh) in detected_eyes:
            eye_regions.append((px + ex, py + ey, ew, eh))
        print(f"   ✓ Found {len(eye_regions)} eye regions")
    
    # STEP 5: Style Matching with SD + IP-Adapter
    print("\n🎨 Step 5: Style matching with Stable Diffusion...")
    print(f"   Using IP-Adapter to copy style from template")
    print(f"   Using ControlNet Tile to preserve details")
    
    result = style_match_inpaint(
        composite_image=composite,
        original_template=original_template,
        person_mask=person_mask,
        style_preset=style_preset,
        custom_prompt=custom_prompt,
        custom_negative=custom_negative,
        denoising_strength=denoising_strength,
        ip_adapter_scale=ip_adapter_scale,
        controlnet_scale=controlnet_scale,
        num_inference_steps=num_steps,
        preserve_eyes=preserve_eyes,
        eye_regions=eye_regions,
        seed=seed,
        use_ip_adapter=use_ip_adapter
    )
    
    print("\n" + "=" * 50)
    print("✅ ART-GEN COMPLETE!")
    print("=" * 50)
    
    return result


# ================== TEMPLATE HELPERS ==================

def get_available_templates():
    templates = []
    if os.path.exists(TEMPLATE_FOLDER):
        for f in os.listdir(TEMPLATE_FOLDER):
            if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS:
                templates.append(f)
    return sorted(templates)


def get_template_path(name=None):
    templates = get_available_templates()
    if not templates:
        return None
    if name:
        if name in templates:
            return os.path.join(TEMPLATE_FOLDER, name)
        for ext in ALLOWED_EXTENSIONS:
            full = f"{name}{ext}"
            if full in templates:
                return os.path.join(TEMPLATE_FOLDER, full)
        return None
    return os.path.join(TEMPLATE_FOLDER, random.choice(templates))


# ================== INIT ==================

try:
    load_u2net_model()
except Exception as e:
    print(f"⚠️ U2Net: {e}")

try:
    load_opencv_cascades()
except Exception as e:
    print(f"⚠️ OpenCV: {e}")

print(f"Templates: {get_available_templates() or 'None (will use default)'}")
print(f"Styles: {list(STYLE_PRESETS.keys())}")


# ================== FLASK ROUTES ==================

@app.route('/')
def home():
    templates = get_available_templates()
    template_opts = ''.join([f'<option value="{t}">{t}</option>' for t in templates])

    # default style for UI
    default_style = "avatar_blue"
    style_opts = ''.join([
        f'<option value="{s}" {"selected" if s == default_style else ""}>{s.replace("_", " ").title()}</option>'
        for s in STYLE_PRESETS.keys()
    ])
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎨 Art-Gen with IP-Adapter</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ 
                font-family: 'Segoe UI', Arial, sans-serif; 
                max-width: 1100px; 
                margin: 0 auto; 
                padding: 20px; 
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); 
                color: #eee; 
                min-height: 100vh;
            }}
            .header {{ text-align: center; padding: 20px 0; }}
            .header h1 {{ 
                font-size: 2.8em; 
                margin: 0;
                background: linear-gradient(90deg, #f857a6, #ff5858);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .header p {{ color: #aaa; font-size: 1.1em; }}
            
            .card {{
                background: rgba(255,255,255,0.05);
                border-radius: 16px;
                padding: 25px;
                margin: 20px 0;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            
            .workflow {{
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                gap: 15px;
                margin: 20px 0;
            }}
            .step {{
                background: rgba(248,87,166,0.1);
                padding: 15px 20px;
                border-radius: 12px;
                text-align: center;
                flex: 1;
                min-width: 150px;
            }}
            .step-num {{
                background: linear-gradient(90deg, #f857a6, #ff5858);
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            
            .form-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            
            .form-group {{ margin: 15px 0; }}
            label {{ 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600; 
                color: #f857a6;
            }}
            
            input, select, textarea {{
                width: 100%;
                padding: 12px;
                border-radius: 8px;
                border: 2px solid rgba(248,87,166,0.3);
                background: rgba(0,0,0,0.3);
                color: #fff;
                font-size: 14px;
                transition: border-color 0.3s;
            }}
            input:focus, select:focus, textarea:focus {{
                outline: none;
                border-color: #f857a6;
            }}
            
            button {{
                background: linear-gradient(90deg, #f857a6, #ff5858);
                color: white;
                padding: 16px 50px;
                border: none;
                border-radius: 30px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }}
            button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 10px 30px rgba(248,87,166,0.4);
            }}
            
            .section-title {{
                color: #ff5858;
                font-size: 1.2em;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid rgba(255,88,88,0.3);
            }}
            
            .tip {{
                background: rgba(248,87,166,0.1);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #f857a6;
                margin: 15px 0;
            }}
            
            .slider-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            input[type="range"] {{
                flex: 1;
                -webkit-appearance: none;
                height: 8px;
                border-radius: 4px;
                background: rgba(248,87,166,0.3);
            }}
            input[type="range"]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #f857a6;
                cursor: pointer;
            }}
            
            .status {{
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
                padding: 15px;
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
            }}
            .status span {{
                padding: 5px 12px;
                background: rgba(248,87,166,0.2);
                border-radius: 20px;
                font-size: 0.9em;
            }}
            
            pre {{
                background: rgba(0,0,0,0.4);
                padding: 20px;
                border-radius: 10px;
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎨 Art-Gen</h1>
            <p>AI-Powered Style Transfer with IP-Adapter & ControlNet</p>
        </div>
        
        <div class="card">
            <div class="workflow">
                <div class="step">
                    <div class="step-num">1</div>
                    <div><strong>U2Net</strong><br>Background Removal</div>
                </div>
                <div class="step">
                    <div class="step-num">2</div>
                    <div><strong>Compose</strong><br>Place on Template</div>
                </div>
                <div class="step">
                    <div class="step-num">3</div>
                    <div><strong>Blue Grade</strong><br>Avatar Style</div>
                </div>
                <div class="step">
                    <div class="step-num">4</div>
                    <div><strong>IP-Adapter</strong><br>Copy Template Style</div>
                </div>
                <div class="step">
                    <div class="step-num">5</div>
                    <div><strong>ControlNet</strong><br>Preserve Details</div>
                </div>
            </div>
        </div>
        
        <div class="status">
            <span>🧠 U2Net: {"✅" if u2net_model else "⏳"}</span>
            <span>🎨 SD+IP-Adapter: {"✅" if sd_pipeline else "⏳ Loads on use"}</span>
            <span>💻 {DEVICE}</span>
        </div>
        
        <form action="/artgen" method="post" enctype="multipart/form-data">
            <div class="card">
                <h3 class="section-title">📷 Input Images</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Person Image:</label>
                        <input type="file" name="image" accept="image/*" required>
                    </div>
                    <div class="form-group">
                        <label>Template (Style Reference):</label>
                        <select name="template">
                            <option value="">Random Template</option>
                            <option value="__default__">Default Gradient</option>
                            {template_opts}
                        </select>
                    </div>
                </div>
                
                <div class="tip">
                    💡 <strong>Avatar Blue</strong> style will make the person blue and match your cinema-style template.
                </div>
            </div>
            
            <div class="card">
                <h3 class="section-title">🎨 Style Settings</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Style Preset:</label>
                        <select name="style">
                            {style_opts}
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Position:</label>
                        <select name="position">
                            <option value="center">Center</option>
                            <option value="bottom-center">Bottom Center</option>
                            <option value="top-center">Top Center</option>
                            <option value="left">Left</option>
                            <option value="right">Right</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label>Denoising Strength (0.3-0.7): <span id="denoise-val">Auto</span></label>
                        <div class="slider-container">
                            <input type="range" name="denoising" min="0.3" max="0.7" step="0.05" value="0.5" 
                                   oninput="document.getElementById('denoise-val').textContent=this.value">
                            <small>Low=subtle, High=more change</small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>IP-Adapter Scale (0.3-0.9): <span id="ip-val">Auto</span></label>
                        <div class="slider-container">
                            <input type="range" name="ip_scale" min="0.3" max="0.9" step="0.05" value="0.7"
                                   oninput="document.getElementById('ip-val').textContent=this.value">
                            <small>Style transfer strength</small>
                        </div>
                    </div>
                </div>
                
                <div class="form-grid">
                    <div class="form-group">
                        <label>ControlNet Scale (0.5-1.0): <span id="cn-val">Auto</span></label>
                        <div class="slider-container">
                            <input type="range" name="cn_scale" min="0.5" max="1.0" step="0.05" value="0.85"
                                   oninput="document.getElementById('cn-val').textContent=this.value">
                            <small>Detail preservation</small>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Scale (Person Size):</label>
                        <input type="number" name="scale" value="0.85" min="0.3" max="1.0" step="0.05">
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3 class="section-title">⚙️ Advanced</h3>
                <div class="form-grid">
                    <div class="form-group">
                        <label>Inference Steps:</label>
                        <input type="number" name="steps" value="30" min="15" max="50">
                    </div>
                    <div class="form-group">
                        <label>Seed (optional):</label>
                        <input type="number" name="seed" placeholder="Random">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>
                        <input type="checkbox" name="preserve_eyes" value="true" checked>
                        👁️ Preserve Original Eyes
                    </label>
                    &nbsp;&nbsp;
                    <label>
                        <input type="checkbox" name="use_ip_adapter" value="true" checked>
                        🎨 Use IP-Adapter (Style Transfer)
                    </label>
                </div>
                
                <div class="form-group">
                    <label>Custom Prompt (optional):</label>
                    <textarea name="custom_prompt" rows="2" placeholder="Override auto-generated prompt..."></textarea>
                </div>
                <div class="form-group">
                    <label>Custom Negative Prompt:</label>
                    <textarea name="custom_negative" rows="2" placeholder="Things to avoid..."></textarea>
                </div>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <button type="submit">🚀 Generate Art</button>
            </div>
        </form>
        
        <div class="card">
            <h3 class="section-title">📡 API</h3>
            <pre>
curl -X POST \\
  -F "image=@person.jpg" \\
  -F "template=your_cinema_template.png" \\
  -F "style=avatar_blue" \\
  -F "denoising=0.55" \\
  -F "ip_scale=0.7" \\
  -F "preserve_eyes=true" \\
  http://localhost:5000/artgen -o result.png
            </pre>
        </div>
    </body>
    </html>
    '''


@app.route('/artgen', methods=['POST'])
def artgen():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Parse params
    template_name = request.form.get('template', '')
    position = request.form.get('position', 'center')
    scale = float(request.form.get('scale', 0.85))
    style = request.form.get('style', 'auto')
    
    denoising = request.form.get('denoising', '').strip()
    denoising = float(denoising) if denoising else None
    
    ip_scale = request.form.get('ip_scale', '').strip()
    ip_scale = float(ip_scale) if ip_scale else None
    
    cn_scale = request.form.get('cn_scale', '').strip()
    cn_scale = float(cn_scale) if cn_scale else None
    
    steps = int(request.form.get('steps', 30))
    seed = request.form.get('seed', '').strip()
    seed = int(seed) if seed else None
    
    preserve_eyes = request.form.get('preserve_eyes', 'true').lower() == 'true'
    use_ip_adapter = request.form.get('use_ip_adapter', 'true').lower() == 'true'
    
    custom_prompt = request.form.get('custom_prompt', '').strip() or None
    custom_negative = request.form.get('custom_negative', '').strip() or None
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        # Get template path
        if template_name == '__default__':
            template_path = None
        elif template_name:
            template_path = get_template_path(template_name)
            if not template_path:
                return jsonify({'error': f'Template not found: {template_name}'}), 404
        else:
            template_path = get_template_path()
        
        # Process
        result = process_artgen(
            person_image=image,
            template_path=template_path,
            position=position,
            scale=scale,
            style_preset=style,
            custom_prompt=custom_prompt,
            custom_negative=custom_negative,
            denoising_strength=denoising,
            ip_adapter_scale=ip_scale,
            controlnet_scale=cn_scale,
            num_steps=steps,
            preserve_eyes=preserve_eyes,
            use_ip_adapter=use_ip_adapter,
            seed=seed
        )
        
        img_io = io.BytesIO()
        result.save(img_io, 'PNG', quality=95)
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/cutout', methods=['POST'])
def cutout():
    """Just background removal"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    erode = int(request.form.get('erode', 3))
    feather = float(request.form.get('feather', 2))
    
    try:
        image = Image.open(file.stream).convert('RGB')
        rgba, _ = remove_background(image, erode_size=erode, feather=feather)
        
        img_io = io.BytesIO()
        rgba.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/styles')
def list_styles():
    return jsonify({name: {
        'denoising': p['denoising'],
        'ip_adapter_scale': p['ip_adapter_scale'],
        'controlnet_scale': p['controlnet_scale'],
        'prompt': p['prompt'][:50] + '...'
    } for name, p in STYLE_PRESETS.items()})


@app.route('/templates')
def list_templates():
    return jsonify({'templates': get_available_templates()})


@app.route('/health')
def health():
    return jsonify({
        'u2net': u2net_model is not None,
        'sd_pipeline': sd_pipeline is not None,
        'ip_adapter': ip_adapter_loaded,
        'device': str(DEVICE)
    })


@app.route('/unload', methods=['POST'])
def unload():
    unload_pipelines()
    return jsonify({'status': 'unloaded'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)