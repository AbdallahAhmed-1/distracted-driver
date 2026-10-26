import json
import numpy as np
import torch
import gradio as gr
from PIL import Image

model = torch.jit.load('model.pt', map_location='cpu').eval()
vocab = json.load(open('vocab.json'))

names = {'c0': 'Safe driving', 'c1': 'Texting - right', 'c2': 'Phone - right',
         'c3': 'Texting - left', 'c4': 'Phone - left', 'c5': 'Operating radio',
         'c6': 'Drinking', 'c7': 'Reaching behind', 'c8': 'Hair/makeup',
         'c9': 'Talking to passenger'}

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img):
    # Matches Resize(224) (center crop). Change this if your item_tfms differ.
    img = img.convert('RGB')
    w, h = img.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    img = img.crop((l, t, l + s, t + s)).resize((224, 224), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255 - mean) / std
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

def predict(img):
    if img is None:
        return None
    with torch.no_grad():
        probs = torch.softmax(model(preprocess(img)), dim=1)[0]
    return {names.get(vocab[i], vocab[i]): float(probs[i]) for i in range(len(vocab))}

with gr.Blocks(title="Distracted Driver Detection") as demo:
    gr.Markdown("# Distracted Driver Detection\nUpload a photo of a driver to classify their behavior.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload image")
        out = gr.Label(num_top_classes=3, label="Output")
    inp.change(predict, inputs=inp, outputs=out)

demo.launch()