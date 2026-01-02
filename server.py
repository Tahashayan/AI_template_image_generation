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
    DPMSolverMultistepScheduler,
)

app = Flask(__name__)

# ================== CONFIG ==================

UPLOAD_FOLDER = 'uploads'
TEMPLATE_FOLDER = 'templates'
MODEL_PATH = 'u2net.pth'

# Model IDs
SD_INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"
CONTROLNET_TILE_ID = "lllyasviel/control_v11f1e_sd15_tile"
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
        "prompt": "same style as reference, matching lighting and colors, seamless blend, professional quality, detailed hands with five fingers",
        "negative": "different style, mismatched colors, obvious edit, artifact, blurry, deformed hands, missing fingers, extra fingers, mutated hands",
        "denoising": 0.5,
        "ip_adapter_scale": 0.6,
        "controlnet_scale": 0.8,
    },
    "avatar_blue": {
        "prompt": (
            "cinematic portrait in a dark movie theater, character lit by strong blue cinematic light, "
            "blue-tinted skin and clothes, monochrome blue color grading, highly detailed, dramatic "
            "lighting, professional photography, Avatar style, perfect hands with five fingers"
        ),
        "negative": (
            "orange lighting, warm skin tones, normal skin color, yellow, green, brown, washed out, "
            "flat lighting, low contrast, artifacts, cartoon, anime, deformed face, extra limbs, "
            "deformed hands, missing fingers, extra fingers, mutated hands, bad anatomy"
        ),
        "denoising": 0.55,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.85,
    },
    "avatar_3d": {
        "prompt": "3D rendered avatar, stylized character, smooth skin, matching environment lighting, CGI quality, Pixar style, perfect hands",
        "negative": "realistic photo, 2d, flat, ugly, deformed, noisy, bad hands, deformed fingers",
        "denoising": 0.6,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "realistic": {
        "prompt": "ultra realistic photo, professional photography, natural skin, matching scene lighting, seamless composite, anatomically correct hands",
        "negative": "cartoon, anime, cgi, artificial, fake looking, obvious edit, deformed hands, wrong fingers",
        "denoising": 0.45,
        "ip_adapter_scale": 0.5,
        "controlnet_scale": 0.85,
    },
    "artistic": {
        "prompt": "artistic portrait, fine art, painterly style, matching artistic environment, museum quality, elegant hands",
        "negative": "photo realistic, plain, boring, low quality, deformed hands",
        "denoising": 0.55,
        "ip_adapter_scale": 0.65,
        "controlnet_scale": 0.75,
    },
    "fantasy": {
        "prompt": "fantasy character, magical atmosphere, ethereal glow, matching mystical environment, enchanted, perfect anatomy",
        "negative": "mundane, realistic, boring, low quality, deformed hands, bad fingers",
        "denoising": 0.65,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "anime": {
        "prompt": "anime style, japanese animation, cel shaded, vibrant colors, matching anime background, proper hand anatomy",
        "negative": "realistic, photo, 3d render, western cartoon, bad hands",
        "denoising": 0.7,
        "ip_adapter_scale": 0.75,
        "controlnet_scale": 0.65,
    },
    "cinematic": {
        "prompt": "cinematic shot, movie quality, dramatic lighting, color graded, matching scene atmosphere, film grain, anatomically correct",
        "negative": "amateur, flat lighting, oversaturated, low budget, deformed hands, missing fingers",
        "denoising": 0.5,
        "ip_adapter_scale": 0.6,
        "controlnet_scale": 0.8,
    },
    "cyberpunk": {
        "prompt": "cyberpunk style, neon lighting, futuristic, holographic effects, matching neon environment, detailed hands",
        "negative": "natural, organic, vintage, old fashioned, deformed hands",
        "denoising": 0.6,
        "ip_adapter_scale": 0.7,
        "controlnet_scale": 0.7,
    },
    "vintage": {
        "prompt": "vintage photo, retro film look, nostalgic colors, film grain, matching period setting, natural hands",
        "negative": "modern, digital, clean, oversaturated, deformed hands",
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
    """Load Stable Diffusion Inpainting with ControlNet Tile and IP-Adapter"""
    global sd_pipeline, controlnet_tile, ip_adapter_loaded
    
    if sd_pipeline is None:
        print("Loading SD Inpainting + ControlNet Tile + IP-Adapter...")
        
        print("  Loading ControlNet Tile...")
        controlnet_tile = ControlNetModel.from_pretrained(
            CONTROLNET_TILE_ID,
            torch_dtype=DTYPE
        )
        
        print("  Loading SD Inpainting Pipeline...")
        sd_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            SD_INPAINT_MODEL_ID,
            controlnet=controlnet_tile,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            sd_pipeline.scheduler.config
        )
        
        sd_pipeline = sd_pipeline.to(DEVICE)
        
        print("  Loading IP-Adapter...")
        sd_pipeline.load_ip_adapter(
            IP_ADAPTER_MODEL_ID,
            subfolder="models",
            weight_name="ip-adapter_sd15.bin"
        )
        ip_adapter_loaded = True
        
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


# ================== HAND DETECTION (OpenCV-based) ==================

def detect_skin_ycrcb(image):
    """
    Detect skin regions using YCrCb color space.
    More robust than HSV for various skin tones.
    """
    img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCrCb)
    
    # Skin color ranges in YCrCb (works for most skin tones)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return skin_mask


def detect_skin_hsv(image):
    """
    Detect skin regions using HSV color space.
    Alternative method for different lighting conditions.
    """
    img_cv = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    
    # Multiple skin color ranges in HSV
    lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
    lower_skin2 = np.array([170, 20, 70], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return skin_mask


def analyze_contour_for_hand(contour, image_shape):
    """
    Analyze if a contour is likely to be a hand based on shape features.
    Uses convexity defects and hull analysis.
    """
    area = cv2.contourArea(contour)
    if area < 500:
        return False, 0
    
    # Get convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return False, 0
    
    # Solidity: ratio of contour area to hull area
    # Hands typically have solidity between 0.5 and 0.9
    solidity = area / hull_area
    
    # Aspect ratio check
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / (h + 1e-6)
    
    # Hands typically have aspect ratios between 0.4 and 2.5
    if aspect < 0.3 or aspect > 3.0:
        return False, 0
    
    # Convexity defects for finger detection
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    defect_count = 0
    if len(hull_indices) > 3 and len(contour) > 3:
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    # d is the depth of the defect
                    if d > 5000:  # Significant defect (potential finger gap)
                        defect_count += 1
        except:
            pass
    
    # Hands typically have 0-5 significant defects (finger gaps)
    # Score based on hand likelihood
    score = 0
    if 0.4 <= solidity <= 0.95:
        score += 30
    if 0.4 <= aspect <= 2.5:
        score += 20
    if 0 <= defect_count <= 6:
        score += 20 + defect_count * 5
    if area > 2000:
        score += 10
    
    return score > 50, score


def detect_hands_opencv(image, person_mask=None, face_regions=None, min_hand_area=800):
    """
    Detect hand regions using multi-method skin detection and shape analysis.
    Excludes face regions to focus on hands only.
    
    Returns: list of (x, y, w, h) bounding boxes for detected hand regions
    """
    img_np = np.array(image.convert('RGB'))
    h, w = img_np.shape[:2]
    
    # Combine YCrCb and HSV skin detection for robustness
    skin_ycrcb = detect_skin_ycrcb(image)
    skin_hsv = detect_skin_hsv(image)
    skin_mask = cv2.bitwise_or(skin_ycrcb, skin_hsv)
    
    # Apply person mask if available
    if person_mask is not None:
        person_mask_np = np.array(person_mask.convert('L').resize((w, h), Image.Resampling.LANCZOS))
        skin_mask = cv2.bitwise_and(skin_mask, person_mask_np)
    
    # Exclude face regions
    if face_regions:
        for (fx, fy, fw, fh) in face_regions:
            # Expand face region to include neck
            expand_x = int(fw * 0.3)
            expand_y_top = int(fh * 0.2)
            expand_y_bottom = int(fh * 0.5)  # More expansion below face for neck
            
            x1 = max(0, fx - expand_x)
            y1 = max(0, fy - expand_y_top)
            x2 = min(w, fx + fw + expand_x)
            y2 = min(h, fy + fh + expand_y_bottom)
            skin_mask[y1:y2, x1:x2] = 0
    
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    hand_regions = []
    hand_scores = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_hand_area:
            continue
        
        # Analyze if this is likely a hand
        is_hand, score = analyze_contour_for_hand(contour, (h, w))
        
        if is_hand or area > min_hand_area * 3:  # Large skin regions are likely hands/arms
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Add generous padding
            padding_x = int(cw * 0.4)
            padding_y = int(ch * 0.4)
            
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(w, x + cw + padding_x)
            y2 = min(h, y + ch + padding_y)
            
            hand_regions.append((x1, y1, x2 - x1, y2 - y1))
            hand_scores.append(score if is_hand else area / 100)
    
    # Merge overlapping regions
    hand_regions = merge_overlapping_regions(hand_regions)
    
    return hand_regions


def detect_arms_by_position(person_mask, face_regions, image_size):
    """
    Detect arm/hand regions based on body position heuristics.
    Uses anatomical proportions to estimate where hands might be.
    """
    if person_mask is None:
        return []
    
    mask_np = np.array(person_mask.convert('L').resize(image_size, Image.Resampling.LANCZOS))
    h, w = mask_np.shape
    
    # Find person bounding box
    rows = np.any(mask_np > 50, axis=1)
    cols = np.any(mask_np > 50, axis=0)
    
    if not rows.any() or not cols.any():
        return []
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    person_height = rmax - rmin
    person_width = cmax - cmin
    person_center_x = (cmin + cmax) // 2
    
    # Exclude face area
    face_bottom = rmin
    if face_regions:
        for (fx, fy, fw, fh) in face_regions:
            face_bottom = max(face_bottom, fy + fh)
    
    arm_regions = []
    
    # Left arm region (typically starts at about 30% body width from center)
    # Arms are usually at 40-80% of body height
    left_arm_x = max(0, cmin - int(person_width * 0.2))
    left_arm_w = int(person_width * 0.5)
    arm_y_start = max(face_bottom, rmin + int(person_height * 0.25))
    arm_y_end = min(rmax, rmin + int(person_height * 0.85))
    
    # Check if there's content in left arm region
    left_region = mask_np[arm_y_start:arm_y_end, left_arm_x:left_arm_x + left_arm_w]
    if np.any(left_region > 50):
        arm_regions.append((left_arm_x, arm_y_start, left_arm_w, arm_y_end - arm_y_start))
    
    # Right arm region
    right_arm_x = person_center_x
    right_arm_w = int(person_width * 0.5) + (cmax - person_center_x)
    
    right_region = mask_np[arm_y_start:arm_y_end, right_arm_x:min(w, right_arm_x + right_arm_w)]
    if np.any(right_region > 50):
        arm_regions.append((right_arm_x, arm_y_start, right_arm_w, arm_y_end - arm_y_start))
    
    return arm_regions


def merge_overlapping_regions(regions, overlap_thresh=0.3):
    """Merge overlapping bounding boxes."""
    if not regions:
        return []
    
    boxes = list(regions)
    merged = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
        
        x1, y1, w1, h1 = boxes[i]
        merge_x1, merge_y1 = x1, y1
        merge_x2, merge_y2 = x1 + w1, y1 + h1
        
        for j in range(len(boxes)):
            if i == j or j in used:
                continue
            
            x2, y2, w2, h2 = boxes[j]
            
            # Check overlap
            ox1 = max(merge_x1, x2)
            oy1 = max(merge_y1, y2)
            ox2 = min(merge_x2, x2 + w2)
            oy2 = min(merge_y2, y2 + h2)
            
            if ox1 < ox2 and oy1 < oy2:
                merge_x1 = min(merge_x1, x2)
                merge_y1 = min(merge_y1, y2)
                merge_x2 = max(merge_x2, x2 + w2)
                merge_y2 = max(merge_y2, y2 + h2)
                used.add(j)
        
        used.add(i)
        merged.append((merge_x1, merge_y1, merge_x2 - merge_x1, merge_y2 - merge_y1))
    
    return merged


def create_protection_mask(image_size, regions, blur_radius=12, shape='rounded'):
    """
    Create a mask to protect specific regions (white = protect, keep original).
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for (x, y, w, h) in regions:
        if shape == 'ellipse':
            draw.ellipse([x, y, x + w, y + h], fill=255)
        elif shape == 'rounded':
            radius = min(w, h) // 4
            draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=255)
        else:
            draw.rectangle([x, y, x + w, y + h], fill=255)
    
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return mask


def create_hand_protection_mask(image_size, hand_regions, blur_radius=15):
    """Create mask specifically for hand protection with extra feathering."""
    return create_protection_mask(image_size, hand_regions, blur_radius=blur_radius, shape='rounded')


def create_eye_protection_mask(image_size, eye_regions, blur_radius=10):
    """Create mask to protect eyes."""
    return create_protection_mask(image_size, eye_regions, blur_radius=blur_radius, shape='ellipse')


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
    
    if template_path and os.path.exists(template_path):
        template = Image.open(template_path).convert('RGBA')
    else:
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
    
    person_arr = np.array(person_rgba)
    alpha = person_arr[:, :, 3]
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    
    if not rows.any() or not cols.any():
        return template.convert('RGB'), original_template, None, None, None, None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    person_cropped = person_rgba.crop((cmin, rmin, cmax + 1, rmax + 1))
    
    target_height = int(template_height * scale)
    aspect = person_cropped.width / person_cropped.height
    target_width = int(target_height * aspect)
    
    max_width = int(template_width * (1 - 2 * padding))
    if target_width > max_width:
        target_width = max_width
        target_height = int(target_width / aspect)
    
    person_resized = person_cropped.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    pad_px = int(template_height * padding)
    positions = {
        'center': ((template_width - target_width) // 2, (template_height - target_height) // 2),
        'top-center': ((template_width - target_width) // 2, pad_px),
        'bottom-center': ((template_width - target_width) // 2, template_height - target_height - pad_px),
        'left': (pad_px, template_height - target_height - pad_px),
        'right': (template_width - target_width - pad_px, template_height - target_height - pad_px),
    }
    x, y = positions.get(position, positions['center'])
    
    composite = template.copy()
    composite.paste(person_resized, (x, y), person_resized)
    
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
    """Apply blue color grade to person region only."""
    if person_mask is None:
        return composite_image

    composite_image = composite_image.convert("RGB")
    w, h = composite_image.size

    mask = person_mask.resize((w, h), Image.Resampling.LANCZOS).convert("L")
    mask_np = np.array(mask).astype(np.float32) / 255.0
    if mask_np.max() < 0.05:
        return composite_image

    img_np = np.array(composite_image)
    mask_3d = np.stack([mask_np] * 3, axis=2)

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_channel, s, v = cv2.split(hsv)

    target_hue = 110.0
    h_blue = (1.0 - strength) * h_channel + strength * target_hue
    s_blue = np.clip(s + strength * 70.0, 0, 255)
    v_blue = np.clip(v + strength * 10.0, 0, 255)

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
    preserve_hands=True,
    eye_regions=None,
    hand_regions=None,
    seed=None,
    use_ip_adapter=True
):
    """
    Use SD Inpainting + IP-Adapter + ControlNet Tile to match person to template style.
    Preserves eyes and hands by blending original pixels back.
    """
    
    preset = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["auto"])
    
    denoising = denoising_strength if denoising_strength is not None else preset["denoising"]
    ip_scale = ip_adapter_scale if ip_adapter_scale is not None else preset["ip_adapter_scale"]
    cn_scale = controlnet_scale if controlnet_scale is not None else preset["controlnet_scale"]
    
    prompt = custom_prompt if custom_prompt else preset["prompt"]
    negative_prompt = custom_negative if custom_negative else preset["negative"]
    
    print(f"🎨 Style: {style_preset}")
    print(f"📝 Prompt: {prompt[:80]}...")
    print(f"⚙️ Denoising: {denoising}, IP-Adapter: {ip_scale}, ControlNet: {cn_scale}")
    
    composite_rgb = composite_image.convert('RGB')
    original_size = composite_rgb.size
    
    w, h = original_size
    max_size = 768
    scale = min(max_size / max(w, h), 1.0)
    new_w = max(64, (int(w * scale) // 8) * 8)
    new_h = max(64, (int(h * scale) // 8) * 8)
    
    composite_resized = composite_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = person_mask.resize((new_w, new_h), Image.Resampling.LANCZOS)
    template_resized = original_template.convert('RGB').resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed) if seed is not None else None
    
    if use_ip_adapter:
        try:
            pipeline = load_sd_pipeline_with_ip_adapter()
            pipeline.set_ip_adapter_scale(ip_scale)
            tile_condition = create_tile_condition(composite_resized, resolution=new_w)
            
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
    
    result = result.resize(original_size, Image.Resampling.LANCZOS)
    
    # === RESTORE ORIGINAL TEMPLATE (background) ===
    result_np = np.array(result).astype(np.float32)
    template_np = np.array(original_template.convert('RGB').resize(original_size)).astype(np.float32)
    composite_np = np.array(composite_image.convert('RGB')).astype(np.float32)
    
    mask_np = np.array(person_mask.convert('L')).astype(np.float32) / 255.0
    mask_3d = np.stack([mask_np] * 3, axis=2)
    
    result_np = result_np * mask_3d + template_np * (1 - mask_3d)
    
    # === PRESERVE EYES ===
    if preserve_eyes and eye_regions:
        print(f"   👁️ Preserving {len(eye_regions)} eye regions")
        eye_mask = create_eye_protection_mask(original_size, eye_regions, blur_radius=8)
        eye_mask_np = np.array(eye_mask).astype(np.float32) / 255.0
        eye_3d = np.stack([eye_mask_np] * 3, axis=2)
        result_np = result_np * (1 - eye_3d) + composite_np * eye_3d
    
    # === PRESERVE HANDS ===
    if preserve_hands and hand_regions:
        print(f"   ✋ Preserving {len(hand_regions)} hand/arm regions")
        hand_mask = create_hand_protection_mask(original_size, hand_regions, blur_radius=15)
        hand_mask_np = np.array(hand_mask).astype(np.float32) / 255.0
        hand_3d = np.stack([hand_mask_np] * 3, axis=2)
        result_np = result_np * (1 - hand_3d) + composite_np * hand_3d
    
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
    preserve_hands=True,
    use_ip_adapter=True,
    erode_size=3,
    feather=2,
    seed=None,
    hand_detection_sensitivity=0.8
):
    """
    Complete Art-Gen Pipeline:
    
    1. Remove background with U2Net
    2. Place person on template
    3. (Optional) Apply blue color grade to person (for avatar_blue style)
    4. Detect eyes and hands (for protection)
    5. Inpaint with IP-Adapter + ControlNet Tile
    6. Restore original eyes and hands
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
    
    # STEP 4: Detect faces, eyes, and hands for preservation
    eye_regions = []
    hand_regions = []
    face_regions = []
    
    if person_resized:
        person_rgb = person_resized.convert('RGB')
        px, py = fg_pos
        
        # Detect faces and eyes
        if preserve_eyes:
            print("\n👁️ Step 4a: Detecting eyes...")
            detected_faces, detected_eyes = detect_faces_and_eyes(person_rgb)
            
            # Offset to composite coordinates
            for (fx, fy, fw, fh) in detected_faces:
                face_regions.append((px + fx, py + fy, fw, fh))
            
            for (ex, ey, ew, eh) in detected_eyes:
                eye_regions.append((px + ex, py + ey, ew, eh))
            
            print(f"   ✓ Found {len(detected_faces)} faces, {len(eye_regions)} eye regions")
        
        # Detect hands
        if preserve_hands:
            print("\n✋ Step 4b: Detecting hands...")
            
            # Create person-only mask for hand detection
            person_alpha = person_resized.split()[3]
            
            # Get face regions relative to person (for exclusion)
            relative_faces = [(fx - px, fy - py, fw, fh) for (fx, fy, fw, fh) in face_regions]
            
            # Detect hands using skin color and shape analysis
            min_hand_area = int(500 * hand_detection_sensitivity)
            detected_hands = detect_hands_opencv(
                person_rgb, 
                person_mask=Image.fromarray(np.array(person_alpha)),
                face_regions=relative_faces,
                min_hand_area=min_hand_area
            )
            
            # If no hands detected by color, try position-based detection
            if not detected_hands:
                print("   ℹ️ Color-based detection found no hands, trying position heuristics...")
                detected_hands = detect_arms_by_position(
                    Image.fromarray(np.array(person_alpha)),
                    relative_faces,
                    (person_resized.width, person_resized.height)
                )
            
            # Offset to composite coordinates
            for (hx, hy, hw, hh) in detected_hands:
                hand_regions.append((px + hx, py + hy, hw, hh))
            
            print(f"   ✓ Found {len(hand_regions)} hand/arm regions")
    
    # STEP 5: Style Matching with SD + IP-Adapter
    print("\n🎨 Step 5: Style matching with Stable Diffusion...")
    print(f"   Using IP-Adapter: {use_ip_adapter}")
    print(f"   Preserve eyes: {preserve_eyes} ({len(eye_regions)} regions)")
    print(f"   Preserve hands: {preserve_hands} ({len(hand_regions)} regions)")
    
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
        preserve_hands=preserve_hands,
        eye_regions=eye_regions,
        hand_regions=hand_regions,
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


# ================== FLASK API ROUTES ==================

@app.route('/artgen', methods=['POST'])
def artgen():
    """
    Main Art-Gen endpoint.
    
    curl example:
    curl -X POST -F "image=@person.jpg" -F "template=template.png" -F "style=avatar_blue" \
         -F "preserve_hands=true" -F "preserve_eyes=true" http://localhost:5000/artgen -o result.png
    """
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
    preserve_hands = request.form.get('preserve_hands', 'true').lower() == 'true'
    use_ip_adapter = request.form.get('use_ip_adapter', 'true').lower() == 'true'
    
    hand_sensitivity = float(request.form.get('hand_sensitivity', 0.8))
    
    custom_prompt = request.form.get('custom_prompt', '').strip() or None
    custom_negative = request.form.get('custom_negative', '').strip() or None
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        if template_name == '__default__':
            template_path = None
        elif template_name:
            template_path = get_template_path(template_name)
            if not template_path:
                return jsonify({'error': f'Template not found: {template_name}'}), 404
        else:
            template_path = get_template_path()
        
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
            preserve_hands=preserve_hands,
            use_ip_adapter=use_ip_adapter,
            seed=seed,
            hand_detection_sensitivity=hand_sensitivity
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
    """
    Background removal only.
    
    curl example:
    curl -X POST -F "image=@person.jpg" http://localhost:5000/cutout -o cutout.png
    """
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


@app.route('/detect_hands', methods=['POST'])
def detect_hands_endpoint():
    """
    Debug endpoint to visualize hand detection.
    
    curl example:
    curl -X POST -F "image=@person.jpg" http://localhost:5000/detect_hands -o hands_debug.png
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    sensitivity = float(request.form.get('sensitivity', 0.8))
    
    try:
        image = Image.open(file.stream).convert('RGB')
        
        # Detect faces first
        faces, eyes = detect_faces_and_eyes(image)
        
        # Detect hands
        min_area = int(500 * sensitivity)
        hands = detect_hands_opencv(image, person_mask=None, face_regions=faces, min_hand_area=min_area)
        
        # Draw detection results
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Draw face regions (blue)
        for (x, y, w, h) in faces:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img_cv, 'Face', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw eye regions (green)
        for (x, y, w, h) in eyes:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw hand regions (red)
        for (x, y, w, h) in hands:
            cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img_cv, 'Hand', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        img_io = io.BytesIO()
        result.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/styles', methods=['GET'])
def list_styles():
    """List available style presets."""
    return jsonify({name: {
        'denoising': p['denoising'],
        'ip_adapter_scale': p['ip_adapter_scale'],
        'controlnet_scale': p['controlnet_scale'],
        'prompt': p['prompt'][:50] + '...'
    } for name, p in STYLE_PRESETS.items()})


@app.route('/templates', methods=['GET'])
def list_templates():
    """List available templates."""
    return jsonify({'templates': get_available_templates()})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'u2net': u2net_model is not None,
        'sd_pipeline': sd_pipeline is not None,
        'ip_adapter': ip_adapter_loaded,
        'device': str(DEVICE),
        'preserve_hands': True,
        'preserve_eyes': True
    })


@app.route('/unload', methods=['POST'])
def unload():
    """Unload SD pipelines to free memory."""
    unload_pipelines()
    return jsonify({'status': 'unloaded'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)