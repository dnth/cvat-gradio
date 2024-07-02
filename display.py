import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from cvat_sdk import make_client
from cvat_sdk.pytorch import TaskVisionDataset, ExtractBoundingBoxes

from cvat_sdk.pytorch import *

# Configuration
CVAT_HOST = 'localhost'
CVAT_USER = 'dnth'
CVAT_PASS = 'dnth'
PORT = 8080
TRAIN_TASK_ID = 5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Initialize CVAT client
client = make_client(CVAT_HOST, port=PORT, credentials=(CVAT_USER, CVAT_PASS))

class CustomTaskVision(TaskVisionDataset):
    def __getitem__(self, sample_index: int):
        sample = self._underlying.samples[sample_index]
        image_path = sample.frame_name
        sample_image = sample.media.load_image()
        sample_target = Target(sample.annotations, self._label_id_to_index)

        if self.transforms:
            sample_image, sample_target = self.transforms(sample_image, sample_target)
        return sample_image, sample_target, image_path

# Initialize dataset
dataset = CustomTaskVision(client, TRAIN_TASK_ID, target_transform=ExtractBoundingBoxes(include_shape_types=['rectangle']))

def plot_image_with_boxes(image, labels):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box in labels['boxes']:
        x, y, x2, y2 = box.tolist()
        w, h = x2 - x, y2 - y
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig

def show_image(index):
    if 0 <= index < len(dataset):
        image, label, img_path = dataset[index]
        fig = plot_image_with_boxes(image, label)
        label_str = f"Boxes: {label['boxes'].tolist()}\nLabels: {label['labels'].tolist()}"
        return fig, label_str, img_path
    else:
        return None, "Invalid index", ""

# Gradio interface
iface = gr.Interface(
    fn=show_image,
    inputs=gr.Number(label="Image Index", precision=0),
    outputs=[gr.Plot(), gr.Textbox(label="Labels", lines=5), gr.Text(label="Image Path")],
    title="PyTorch Dataset Image Viewer",
    description="Enter an index to view the corresponding image from the dataset."
)

iface.launch()