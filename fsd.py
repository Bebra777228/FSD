import numpy as np
import gradio as gr
import requests
import time
import json
import base64
import os
from io import BytesIO
import PIL
from PIL.ExifTags import TAGS
import html
import re
import torch
from PIL import Image

# Устанавливаем устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key,
            "Content-Type": "application/json"
        }

    def _request(self, method, url, params=None):
        response = requests.request(method, url, headers=self.headers, json=params)
        if response.status_code != 200:
            raise Exception(f"Плохой ответ Prodia: {response.status_code}")
        return response.json()

    def generate_sd(self, params):
        return self._request("POST", f"{self.base}/sd/generate", params)

    def generate_xl(self, params):
        return self._request("POST", f"{self.base}/sdxl/generate", params)

    def upscale(self, params):
        return self._request("POST", f"{self.base}/sd/upscale", params)

    def get_job(self, job_id):
        return self._request("GET", f"{self.base}/job/{job_id}")

    def wait(self, job):
        job_result = job
        start_wait = time.time()

        while job_result['status'] not in ['succeeded', 'failed']:
            if time.time() - start_wait > 100:
                raise Exception(f"Ошибка! Долгая генерация: {job_result['status']}")
            time.sleep(0.25)
            job_result = self.get_job(job['job'])

        return job_result

    def list_models_sd(self):
        return self._request("GET", f"{self.base}/sd/models")

    def list_models_xl(self):
        return self._request("GET", f"{self.base}/sdxl/models")

    def list_samplers_sd(self):
        return self._request("GET", f"{self.base}/sd/samplers")

    def list_samplers_xl(self):
        return self._request("GET", f"{self.base}/sdxl/samplers")

    def list_loras_sd(self):
        return self._request("GET", f"{self.base}/sd/loras")

    def list_loras_xl(self):
        return self._request("GET", f"{self.base}/sdxl/loras")

def remove_id_and_ext(text):
    """Remove ID and extension from the model name."""
    text = re.sub(r'\[.*\]$', '', text)
    if text.endswith("safetensors"):
        text = text[:-13]
    elif text.endswith("ckpt"):
        text = text[:-4]
    return text

def place_lora(current_prompt, lora_name):
    """Place LoRA in the prompt."""
    pattern = rf"<lora:{lora_name}:.*?>"
    if re.search(pattern, current_prompt):
        return re.sub(pattern, "", current_prompt)
    else:
        return current_prompt + f"<lora:{lora_name}:1>"

def txt2img_sd(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed, upscale):
    """Generate image using Stable Diffusion."""
    result = prodia_client.generate_sd({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
        "upscale": upscale
    })

    job = prodia_client.wait(result)

    if job['status'] != "succeeded":
        raise gr.Error("job failed")

    return job["imageUrl"]

def txt2img_xl(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
    """Generate image using Stable Diffusion XL."""
    result = prodia_client.generate_xl({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed,
    })

    job = prodia_client.wait(result)

    if job['status'] != "succeeded":
        raise gr.Error("job failed")

    return job["imageUrl"]

def get_exif_data(image):
    """Get EXIF data from the image."""
    items = image.info
    info = ''
    for key, text in items.items():
        info += (
            f"""
            <div>
                <p><b>{plaintext_to_html(str(key))}</b></p>
                <p>{plaintext_to_html(str(text))}</p>
            </div>
            """
        ).strip() + "\n"

    if not info:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return info

def plaintext_to_html(text, classname=None):
    """Convert plaintext to HTML."""
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))
    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"

# Initialize Prodia clients
prodia_client = Prodia(api_key=os.getenv("PRODIA_API_KEY"))

# Fetch model and LoRA lists
model_list_sd = prodia_client.list_models_sd()
lora_list_sd = prodia_client.list_loras_sd()
model_names = {remove_id_and_ext(model_name): model_name for model_name in model_list_sd}

model_list_xl = prodia_client.list_models_xl()
lora_list_xl = prodia_client.list_loras_xl()
model_names_xl = {remove_id_and_ext(model_name): model_name for model_name in model_list_xl}

# Test parameters
test_params = {
    "prompt": "test",
    "negative_prompt": "",
    "model": "absolutereality_v181.safetensors [3d9d4d2b]",
    "steps": 1,
    "sampler": "DPM++ 2M Karras",
    "cfg_scale": 1,
    "width": 8,
    "height": 8,
    "seed": -1,
    "upscale": False
}

# Run test
txt2img_sd(**test_params)

# CSS for Gradio interface
css = """
#generate {
    height: 100%;
}

.gr-container img {
    display: block;
    margin: 0 auto;
}
"""

# Gradio interface
with gr.Blocks(css=css, theme=gr.themes.Soft(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="neutral",
    spacing_size="sm",
    radius_size="lg",
)) as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Fast Stable Diffusion", id='t2i'):
            with gr.Row():
                with gr.Column(scale=6):
                    model = gr.Dropdown(interactive=True, value="absolutereality_v181.safetensors [3d9d4d2b]", show_label=True, label="Models:", choices=model_list_sd)

            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=6, min_width=600):
                        prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Prompt", show_label=False, lines=3)
                        negative_prompt = gr.Textbox(placeholder="Negative Prompt", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
                    with gr.Column():
                        text_button = gr.Button("Generate", variant='primary', elem_id="generate")

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tab("Generation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Sampling Method", choices=prodia_client.list_samplers_sd())
                            with gr.Column(scale=1):
                                steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

                        with gr.Row():
                            with gr.Column(scale=1):
                                width = gr.Slider(label="Width", minimum=8, maximum=1024, value=512, step=8)
                                height = gr.Slider(label="Height", minimum=8, maximum=1024, value=512, step=8)
                                upscale = gr.Checkbox(label="Up-Scale Image x2", scale=1)

                        cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, value=7, step=1)
                        seed = gr.Number(label="Seed", value=-1)

                    with gr.Tab("Lora"):
                        with gr.Row():
                            for lora in lora_list_sd:
                                lora_btn = gr.Button(lora, size="sm")
                                lora_btn.click(place_lora, inputs=[prompt, lora_btn], outputs=prompt)

                with gr.Column(scale=2):
                    image_output = gr.Image(value="https://images.prodia.xyz/8ede1a7c-c0ee-4ded-987d-6ffed35fc477.png")

            text_button.click(txt2img_sd, inputs=[prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed, upscale], outputs=image_output, concurrency_limit=64)

        with gr.Tab("Fast Stable Diffusion XL"):
            with gr.Row():
                with gr.Column(scale=3):
                    xl_model = gr.Dropdown(interactive=True, value="sd_xl_base_1.0.safetensors [be9edd61]", show_label=True, label="Models:", choices=model_list_xl)

            with gr.Group():
                with gr.Row(equal_height=True):
                    with gr.Column(scale=6, min_width=600):
                        xl_prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Prompt", show_label=False, lines=3)
                        xl_negative_prompt = gr.Textbox(placeholder="Negative Prompt", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
                    with gr.Column():
                        xl_text_button = gr.Button("Generate", variant='primary', elem_id="generate")

            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tab("Generation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                xl_sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Sampling Method", choices=prodia_client.list_samplers_xl())
                            with gr.Column(scale=1):
                                xl_steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)

                        with gr.Row():
                            with gr.Column(scale=1):
                                xl_width = gr.Slider(label="Width", minimum=8, maximum=1024, value=512, step=8)
                                xl_height = gr.Slider(label="Height", minimum=8, maximum=1024, value=512, step=8)

                        xl_cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, value=7, step=1)
                        xl_seed = gr.Number(label="Seed", value=-1)

                    with gr.Tab("Lora"):
                        with gr.Row():
                            for lora in lora_list_xl:
                                lora_btn = gr.Button(lora, size="sm")
                                lora_btn.click(place_lora, inputs=[prompt, lora_btn], outputs=prompt)

                with gr.Column(scale=2):
                    xl_image_output = gr.Image(value="https://cdn-uploads.huggingface.co/production/uploads/noauth/XWJyh9DhMGXrzyRJk7SfP.png")

            xl_text_button.click(txt2img_xl, inputs=[xl_prompt, xl_negative_prompt, xl_model, xl_steps, xl_sampler, xl_cfg_scale, xl_width, xl_height, xl_seed], outputs=xl_image_output)

        with gr.Tab("Image Info"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil")

                with gr.Column():
                    exif_output = gr.HTML(label="EXIF Data")

            image_input.upload(get_exif_data, inputs=[image_input], outputs=exif_output)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

demo.launch(share=True, show_api=False)
