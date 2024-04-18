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
from RealESRGAN import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model2 = RealESRGAN(device, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
if torch.cuda.device_count() > 1:
    model2 = torch.nn.DataParallel(model2)

model4 = RealESRGAN(device, scale=4)
model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
if torch.cuda.device_count() > 1:
    model4 = torch.nn.DataParallel(model4)

model8 = RealESRGAN(device, scale=8)
model8.load_weights('weights/RealESRGAN_x8.pth', download=True)
if torch.cuda.device_count() > 1:
    model8 = torch.nn.DataParallel(model8)

def inference(image, size):
    global model2
    global model4
    global model8
    if image is None:
        raise gr.Error("Изображение не загружено")

    width, height = image.size
    if width >= 5000 or height >= 5000:
        raise gr.Error("Изображение слишком большое.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if size == '2x':
        try:
            result = model2.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model2 = RealESRGAN(device, scale=2)
            model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        try:
            result = model4.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model4 = RealESRGAN(device, scale=4)
            model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
            result = model4.predict(image.convert('RGB'))
    else:
        try:
            result = model8.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model8 = RealESRGAN(device, scale=8)
            model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
            result = model8.predict(image.convert('RGB'))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Размер изображения ({device}): {size} ... OK")
    return result

class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key
        }

    def generate(self, params, model_type='sd'):
        if model_type not in ['sd', 'sdxl']:
            raise ValueError("Неверный тип модели. Используйте 'sd' или 'sdxl'.")

        endpoint = f"{self.base}/{model_type}/generate"
        response = self._post(endpoint, params)
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

    def list_models(self, model_type='sd'):
        if model_type not in ['sd', 'sdxl']:
            raise ValueError("Неверный тип модели. Используйте 'sd' или 'sdxl'.")

        endpoint = f"{self.base}/{model_type}/models"
        response = self._get(endpoint)
        return response.json()

    def list_samplers(self, model_type='sd'):
        if model_type not in ['sd', 'sdxl']:
            raise ValueError("Неверный тип модели. Используйте 'sd' или 'sdxl'.")

        endpoint = f"{self.base}/{model_type}/samplers"
        response = self._get(endpoint)
        return response.json()

    def _post(self, url, params):
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(params))

        if response.status_code != 200:
            raise Exception(f"Плохой ответ Prodia: {response.status_code}")

        return response

    def _get(self, url):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Плохой ответ Prodia: {response.status_code}")

        return response

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

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
    result = {tabs: gr.update(selected="t2i")}
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
        return result

prodia_client = Prodia(api_key=os.getenv("PRODIA_API_KEY"))

model_list = prodia_client.list_models()
model_names = {}
for model_name in model_list:
    name_without_ext = remove_id_and_ext(model_name)
    model_names[name_without_ext] = model_name

model_list_xl = prodia_client.list_models(model_type='sdxl')
model_names_xl = {}
for model_name in model_list_xl:
    name_without_ext = remove_id_and_ext(model_name)
    model_names_xl[name_without_ext] = model_name

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

def flip_text(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
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
    }, model_type='sdxl')

    job = prodia_client.wait(result)

    return job["imageUrl"]

css = """
#generate {
    height: 100%;
}

.gr-container img {
    display: block;
    margin: 0 auto;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Велком"):
            gr.HTML("""
            <div style="padding: 20px;">
                <h1 style="text-align: center;">🎨 Веб-интерфейс Fast Stable Diffusion от Politrees</h1>
                <p style="text-align: center;">Создавайте изображения с помощью стабильных диффузионных моделей!</p>

                <hr style="margin: 20px 0;">

                <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="flex: 1; margin-bottom: 10px;">
                        <h2 style="text-align: center;">Fast Stable Diffusion</h2>
                        <img src="https://example.com/fast_stable_diffusion.png" alt="Fast Stable Diffusion" width="100%">
                        <p style="text-align: center;">Не алмаз, а золото. Лучший вариант для генерации изображений.</p>
                    </div>
                    <div style="flex: 1; margin-bottom: 10px;">
                        <h2 style="text-align: center;">Fast Stable Diffusion XL</h2>
                        <img src="https://example.com/fast_stable_diffusion_xl.png" alt="Fast Stable Diffusion XL" width="100%">
                        <p style="text-align: center;">То ли гений, то ли псих. Автор нейронки наверное был под ЛСД когда ее делал.</p>
                    </div>
                    <div style="flex: 1; margin-bottom: 10px;">
                        <h2 style="text-align: center;">Улучшение изображений</h2>
                        <img src="https://example.com/real_esrgan.png" alt="Real-ESRGAN" width="100%">
                        <p style="text-align: center;">Улучшайте качество ваших изображений с помощью моделей RealESRGAN.</p>
                    </div>
                    <div style="flex: 1; margin-bottom: 10px;">
                        <h2 style="text-align: center;">Информация о сгенерированном изображении</h2>
                        <img src="https://example.com/png_info.png" alt="PNG Info" width="100%">
                        <p style="text-align: center;">Извлекайте данные из изображений для повторной генерации (prompt, negative prompt и др.).</p>
                    </div>
                </div>

                <hr style="margin: 20px 0;">

                <p style="text-align: center;">Экспериментируйте и создавайте!</p>
            </div>
            """)


        with gr.Tab("Fast Stable Diffusion", id='t2i'):
            with gr.Row():
                with gr.Column(scale=6, min_width=600):
                    prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Напишите тут то что хочешь сгенерировать", show_label=False, lines=3)
                    negative_prompt = gr.Textbox(placeholder="Напишите тут то что надо убрать из изображения", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
                with gr.Column():
                    text_button = gr.Button("Генерировать", variant='primary', elem_id="generate")

            with gr.Row():
                with gr.Column(scale=3):
                        model = gr.Dropdown(interactive=True, value="absolutereality_v181.safetensors [3d9d4d2b]", show_label=True, label="Модели:", choices=model_list)

                        with gr.Row():
                            with gr.Column(scale=1):
                                sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Метод генерации:", choices=prodia_client.list_samplers())

                            with gr.Column(scale=1):
                                steps = gr.Slider(label="Количество шагов генерации", minimum=1, maximum=100, value=20, step=1)

                        with gr.Row():
                            with gr.Column(scale=1):
                                width = gr.Slider(label="Ширина изображения", minimum=128, maximum=1024, value=512, step=8)
                                height = gr.Slider(label="Высота изображения", minimum=128, maximum=1024, value=512, step=8)

                        cfg_scale = gr.Slider(label="Масштаб CFG", minimum=1, maximum=30, value=7, step=1)
                        seed = gr.Number(label="Seed", value=-1)

                with gr.Column(scale=2):
                    image_output = gr.Image(value="https://images.prodia.xyz/8ede1a7c-c0ee-4ded-987d-6ffed35fc477.png")

            text_button.click(txt2img, inputs=[prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed], outputs=image_output, concurrency_limit=64)

        with gr.Tab("Fast Stable Diffusion XL"):
            with gr.Row():
                with gr.Column(scale=6, min_width=600):
                    xl_prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Напишите тут то что хочешь сгенерировать.", show_label=False, lines=3)
                    xl_negative_prompt = gr.Textbox(placeholder="Напишите тут то что надо убрать из изображения", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
                with gr.Column():
                    xl_text_button = gr.Button("Генерировать", variant='primary', elem_id="generate")

            with gr.Row():
                with gr.Column(scale=3):
                        xl_model = gr.Dropdown(interactive=True, value="sd_xl_base_1.0.safetensors [be9edd61]", show_label=True, label="Модели:", choices=model_list_xl)

                        with gr.Row():
                            with gr.Column(scale=1):
                                xl_sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Метод генерации:", choices=prodia_client.list_samplers(model_type='sdxl'))

                            with gr.Column(scale=1):
                                xl_steps = gr.Slider(label="Количество шагов генерации", minimum=1, maximum=100, value=20, step=1)

                        with gr.Row():
                            with gr.Column(scale=1):
                                xl_width = gr.Slider(label="Ширина изображения", minimum=128, maximum=1024, value=512, step=8)
                                xl_height = gr.Slider(label="Высота изображения", minimum=128, maximum=1024, value=512, step=8)

                        xl_cfg_scale = gr.Slider(label="Масштаб CFG", minimum=1, maximum=30, value=7, step=1)
                        xl_seed = gr.Number(label="Seed", value=-1)

                with gr.Column(scale=2):
                    xl_image_output = gr.Image(value="https://cdn-uploads.huggingface.co/production/uploads/noauth/XWJyh9DhMGXrzyRJk7SfP.png")

            xl_text_button.click(flip_text, inputs=[xl_prompt, xl_negative_prompt, xl_model, xl_steps, xl_sampler, xl_cfg_scale, xl_width, xl_height, xl_seed], outputs=xl_image_output)

        with gr.Tab("Улучшение изображений"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil")

                with gr.Column():
                    improved_image_output = gr.Image(type="pil")
                    improve_image_btn = gr.Button("Улучшить изображение")

            improve_image_btn.click(inference, inputs=[image_input, gr.Radio(['2x', '4x', '8x'], type="value", value='2x', label='Увеличение разрешения изображения')], outputs=improved_image_output)

        with gr.Tab("Информация о сгенерированном изображении"):
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
                    message = "Ничего не найдено в изображении."
                    info = f"<div><p>{message}<p></div>"

                return info

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil")

                with gr.Column():
                    exif_output = gr.HTML(label="EXIF Data")
                    send_to_txt2img_btn = gr.Button("Показать информацию")

            image_input.upload(get_exif_data, inputs=[image_input], outputs=exif_output)
            send_to_txt2img_btn.click(send_to_txt2img, inputs=[image_input], outputs=[tabs, prompt, negative_prompt, steps, seed, model, sampler, width, height, cfg_scale], concurrency_limit=64)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

os.system('clear_output()')

demo.queue(max_size=80, api_open=False).launch(max_threads=256, share=True, show_api=False)
