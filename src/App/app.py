import os
from random import random
import sys
import gradio as gr
import streamlit as st

import torch
import torch.nn as nn

import Models.upscaling.resolution as resolution
from Models.basic.models import BasicDiscriminator, BasicGenerator
from Tools.utils import color_histogram_mapping, denormalize

# we'll use gradio to render the front-end app
# and the app will run on the hugging face server 
# reference: https://huggingface.co/spaces/akhaliq/JoJoGAN/blob/main/app.py
# reference: https://huggingface.co/spaces/therealcyberlord/abstract-art-generation

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")

# title = "title"
# article = "Demo version of GAN-Sonmi"

# interface = gr.Interface(
#     title,
#     article,
#     allow_flagging="never",
# )

# interface.launch()

latent_size = 100 
checkpoint_path =  r'../add/some/checkpoint/path.chkpt'

st.title("침착한 이미지 생성기")
use_srgan = st.checkbox("apply image enhancement", ('Yes', 'No'))

@st.cache(allow_output_mutation=True)
def load_dcgan():
    model = torch.jit.load(checkpoint_path, map_location=device)
    return model

@st.cache(allow_output_mutation=True)
def load_enhancer():
    model = torch.jit.load(checkpoint_path, map_location=device)
    return model


PRETRAINED_PATH = './src/Models/pretrained'
def load_GFPGAN(use=False):
    model_name = 'GFPGANv1.3'
    gfpgan_dir = './src/Models/pretrained/gfpgan'
    path = os.path.join(PRETRAINED_PATH, model_name + '.pth')

    if not os.path.isfile(path):
        raise Exception(f"GFPGAN model not found at '{path}'")

    if use:
        return True 
    
    if os.path.exists(gfpgan_dir):
        sys.path.append(os.path.abspath(gfpgan_dir))
    else:
        raise Exception(f"GFPGAN directory not found at '{gfpgan_dir}'")

num_images = st.slider("Number of images", 1, 10, 3)
generate = st.sidebar.button("Generate")



if generate:
    torch.manual_seed(42)
    random.seed(42)

    z = torch.randn(1, latent_size, 1, 1).to(device)
    generator = load_dcgan()
    generator.eval()

    with torch.no_grad():
        fakes = generator(z).detach()

    if use_srgan == 'Yes':
        # restore to the checkpoint
        enhancer = resolution.GeneratorRRDB(channels=1, filters=64, num_res_blocks=23).to(device)
        enhancer_chkpt = load_enhancer()
        enhancer.load_state_dict(enhancer_chkpt)

        enhancer.eval()
        with torch.no_grad():
            enhanced_fakes = enhancer(fakes).detach().cpu()
        
        color_grading = color_histogram_mapping(enhanced_fakes, fakes.cpu())

        cols = st.columns(num_images)
        for i in range(color_grading):
            cols[i].image(denormalize(
                color_grading[i]).permute(1, 2, 0).numpy(), 
                use_column_width=True,
            )
    
    if use_srgan == "No":
        fakes = fakes.cpu()

        cols = st.columns(num_images)
        for i in range(len(fakes)):
            cols[i].image(
                denormalize(fakes[i]).permute(1, 2, 0).numpy(), 
                use_column_width=True,
            )

