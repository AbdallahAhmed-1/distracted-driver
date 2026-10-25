from fastai.vision.all import *
import gradio as gr
import torch
import pathlib, sys
sys.modules['pathlib._local'] = pathlib

learn = load_learner('export.pkl')
learn.to_fp32()                      # safe for CPU inference
labels = learn.dls.vocab

names = {'c0': 'Safe driving', 'c1': 'Texting - right', 'c2': 'Phone - right',
         'c3': 'Texting - left', 'c4': 'Phone - left', 'c5': 'Operating radio',
         'c6': 'Drinking', 'c7': 'Reaching behind', 'c8': 'Hair/makeup',
         'c9': 'Talking to passenger'}

@spaces.GPU
def predict(img):
    if img is None:
        return None
    pred, idx, probs = learn.predict(PILImage.create(img))
    return {names.get(str(labels[i]), str(labels[i])): float(probs[i])
            for i in range(len(labels))}

with gr.Blocks(title="Distracted Driver Detection") as demo:
    gr.Markdown("# Distracted Driver Detection\nUpload a photo of a driver to classify their behavior.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload image")
        out = gr.Label(num_top_classes=3, label="Output")
    inp.change(predict, inputs=inp, outputs=out)

demo.launch()
# h