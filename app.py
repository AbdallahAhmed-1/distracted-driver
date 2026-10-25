from fastai.vision.all import *
import gradio as gr


learn = load_learner('export.pkl')
labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=5),
    examples=['example1.jpg'],  # optional; remove if you have none
)
demo.launch()