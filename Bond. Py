import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

Model config
model_name = "CompVis/stable-diffusion-v1-4"
use_auth_token = "hf_QzbKVLPMYjJRvxdMZDyVrgcPfLPpXxbH"

Prompts
prompt = """
A hyper-realistic woman, 20 yrs, pale skin, blue undertones,
platinum blonde hair, plush lips, perfect body, long hair,
cat ears, tail, 
#{outfit},
sugar skull makeup, earrings, tattoo, 
looking at viewer with #{expression}, 
large breasts, red eyes, 
#{pose} angle
"""

Outfit, expression, pose options
outfit_option = "black lace dress"
expression_option = "soft smile"
pose_option = "direct gaze"

prompt = prompt.replace("#{outfit}", outfit_option)
prompt = prompt.replace("#{expression}", expression_option)
prompt = prompt.replace("#{pose}", pose_option)

Parameters
width, height = 10240, 5760 # HIGH RESOLUTION
num_inference_steps = 300 
guidance_scale = 10 
torch_dtype = torch.float16
revision = "fp16"

Generate image
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    revision=revision,
    torch_dtype=torch_dtype,
    use_auth_token=use_auth_token
)

image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale
).images[0]

Save image
image.save("platinum_blonde_high_res.png")
