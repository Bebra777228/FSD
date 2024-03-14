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


class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key
        }
    
    def generate(self, params):
        response = self._post(f"{self.base}/sd/generate", params)
        return response.json()
    
    def transform(self, params):
        response = self._post(f"{self.base}/sd/transform", params)
        return response.json()
    
    def controlnet(self, params):
        response = self._post(f"{self.base}/sd/controlnet", params)
        return response.json()
    
    def get_job(self, job_id):
        response = self._get(f"{self.base}/job/{job_id}")
        return response.json()

    def wait(self, job):
        job_result = job

        while job_result['status'] not in ['succeeded', 'failed']:
            time.sleep(0.25)
            job_result = self.get_job(job['job'])

        return job_result

    def list_models(self):
        response = self._get(f"{self.base}/sd/models")
        return response.json()

    def list_samplers(self):
        response = self._get(f"{self.base}/sd/samplers")
        return response.json()

    def _post(self, url, params):
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(params))

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response

    def _get(self, url):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response


def image_to_base64(image):
    # Convert the image to bytes
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can change format to PNG if needed
    
    # Encode the bytes to base64
    img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')  # Convert bytes to string

def remove_id_and_ext(text):
    text = re.sub(r'\[.*\]$', '', text)
    extension = text[-12:].strip()
    if extension == "safetensors":
        text = text[:-13]
    elif extension == "ckpt":
        text = text[:-4]
    return text

def get_data(text):
    results = {}
    patterns = {
        'prompt': r'(.*)',
        'negative_prompt': r'Negative prompt: (.*)',
        'steps': r'Steps: (\d+),',
        'seed': r'Seed: (\d+),',
        'sampler': r'Sampler:\s*([^\s,]+(?:\s+[^\s,]+)*)', 
        'model': r'Model:\s*([^\s,]+)',
        'cfg_scale': r'CFG scale:\s*([\d\.]+)',
        'size': r'Size:\s*([0-9]+x[0-9]+)'
        }
    for key in ['prompt', 'negative_prompt', 'steps', 'seed', 'sampler', 'model', 'cfg_scale', 'size']:
        match = re.search(patterns[key], text)
        if match:
            results[key] = match.group(1)
        else:
            results[key] = None
    if results['size'] is not None:
        w, h = results['size'].split("x")
        results['w'] = w
        results['h'] = h
    else:
        results['w'] = None
        results['h'] = None
    return results

def send_to_txt2img(image):
    
    result = {tabs: gr.Tabs.update(selected="t2i")}

    try:
        text = image.info['parameters']
        data = get_data(text)
        result[prompt] = gr.update(value=data['prompt'])
        result[negative_prompt] = gr.update(value=data['negative_prompt']) if data['negative_prompt'] is not None else gr.update()
        result[steps] = gr.update(value=int(data['steps'])) if data['steps'] is not None else gr.update()
        result[seed] = gr.update(value=int(data['seed'])) if data['seed'] is not None else gr.update()
        result[cfg_scale] = gr.update(value=float(data['cfg_scale'])) if data['cfg_scale'] is not None else gr.update()
        result[width] = gr.update(value=int(data['w'])) if data['w'] is not None else gr.update()
        result[height] = gr.update(value=int(data['h'])) if data['h'] is not None else gr.update()
        result[sampler] = gr.update(value=data['sampler']) if data['sampler'] is not None else gr.update()
        if model in model_names:
            result[model] = gr.update(value=model_names[model])
        else:
            result[model] = gr.update()
        return result

    except Exception as e:
        print(e)
        result[prompt] = gr.update()
        result[negative_prompt] = gr.update()
        result[steps] = gr.update()
        result[seed] = gr.update()
        result[cfg_scale] = gr.update()
        result[width] = gr.update()
        result[height] = gr.update()
        result[sampler] = gr.update()
        result[model] = gr.update()

        return result


prodia_client = Prodia(api_key=os.getenv("PRODIA_API_KEY"))
model_list = prodia_client.list_models()
model_names = {}

for model_name in model_list:
    name_without_ext = remove_id_and_ext(model_name)
    model_names[name_without_ext] = model_name

def txt2img(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
    result = prodia_client.generate({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed
    })

    job = prodia_client.wait(result)

    return job["imageUrl"]


css = """
#generate {
    height: 100%;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=6):
            model = gr.Dropdown(interactive=True,value="absolutereality_v181.safetensors [3d9d4d2b]", show_label=True, label="Stable Diffusion Checkpoint", choices=prodia_client.list_models())
  

    with gr.Tabs() as tabs:
        with gr.Tab("txt2img", id='t2i'):
            with gr.Row():
                with gr.Column(scale=6, min_width=600):
                    prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Prompt", show_label=False, lines=3)
                    negative_prompt = gr.Textbox(placeholder="Negative Prompt", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
                with gr.Column():
                    text_button = gr.Button("Generate", variant='primary', elem_id="generate")
                    
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Tab("Generation"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Sampling Method", choices=prodia_client.list_samplers())
                                
                            with gr.Column(scale=1):
                                steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=100, value=25, step=1)
    
                        with gr.Row():
                            with gr.Column(scale=1):
                                width = gr.Slider(label="Width", minimum=128, maximum=1024, value=512, step=8)
                                height = gr.Slider(label="Height", minimum=128, maximum=1024, value=512, step=8)
    
                        cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=30, value=7, step=1)
                        seed = gr.Number(label="Seed", value=-1)
    
                    
                with gr.Column(scale=2):
                    image_output = gr.Image(value="https://images.prodia.xyz/8ede1a7c-c0ee-4ded-987d-6ffed35fc477.png")
    
            text_button.click(txt2img, inputs=[prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed], outputs=image_output)
        
        
        with gr.Tab("PNG Info"):
            def plaintext_to_html(text, classname=None):
                content = "<br>\n".join(html.escape(x) for x in text.split('\n'))
    
                return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"
    
    
            def get_exif_data(image):
                items = image.info
    
                info = ''
                for key, text in items.items():
                    info += f"""
                    <div>
                    <p><b>{plaintext_to_html(str(key))}</b></p>
                    <p>{plaintext_to_html(str(text))}</p>
                    </div>
                    """.strip()+"\n"
    
                if len(info) == 0:
                    message = "Nothing found in the image."
                    info = f"<div><p>{message}<p></div>"
    
                return info
    
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil")
                    
                with gr.Column():
                    exif_output = gr.HTML(label="EXIF Data")
                    send_to_txt2img_btn = gr.Button("Send to txt2img")
    
            image_input.upload(get_exif_data, inputs=[image_input], outputs=exif_output)
            send_to_txt2img_btn.click(send_to_txt2img, inputs=[image_input], outputs=[tabs, prompt, negative_prompt, steps, seed,
                                                                                          model, sampler, width, height, cfg_scale])

demo.queue(max_size=1024).launch(max_threads=512, share=True)