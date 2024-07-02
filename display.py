import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from cvat_sdk import make_client
from cvat_sdk.pytorch import TaskVisionDataset, ExtractBoundingBoxes
from cvat_sdk.pytorch import Target

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

class CustomTaskVision(TaskVisionDataset):
    def __getitem__(self, sample_index: int):
        sample = self._underlying.samples[sample_index]
        image_path = sample.frame_name
        sample_image = sample.media.load_image()
        sample_target = Target(sample.annotations, self._label_id_to_index)

        if self.transforms:
            sample_image, sample_target = self.transforms(sample_image, sample_target)
        return sample_image, sample_target, image_path

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

def connect_and_load_dataset(host, user, password, port, task_id):
    try:
        client = make_client(host, port=int(port), credentials=(user, password))
        dataset = CustomTaskVision(client, int(task_id), target_transform=ExtractBoundingBoxes(include_shape_types=['rectangle']))
        return dataset, "Connected successfully"
    except Exception as e:
        return None, f"Connection failed: {str(e)}"

def show_image(dataset, index):
    if dataset is None:
        return None, "Please connect to CVAT first", ""
    if 0 <= index < len(dataset):
        image, label, img_path = dataset[index]
        fig = plot_image_with_boxes(image, label)
        label_str = f"Boxes: {label['boxes'].tolist()}\nLabels: {label['labels'].tolist()}"
        return fig, label_str, img_path
    else:
        return None, "Invalid index", ""

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# PyTorch Dataset Image Viewer")
    gr.Markdown("Enter CVAT connection details and connect before viewing images.")
    
    with gr.Row():

        with gr.Column():
            host = gr.Textbox(label="CVAT Host", value="localhost")
            port = gr.Textbox(label="CVAT Port", value="8080")
            user = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            task_id = gr.Textbox(label="Task ID")
        
            connect_btn = gr.Button("Connect")
        
        connection_status = gr.Textbox(label="Connection Status")
    
    with gr.Row():
        index_input = gr.Number(label="Image Index", precision=0)
        view_btn = gr.Button("View Image")
    
    image_output = gr.Plot()
    label_output = gr.Textbox(label="Labels", lines=5)
    path_output = gr.Text(label="Image Path")
    
    dataset = gr.State(None)
    
    connect_btn.click(
        connect_and_load_dataset,
        inputs=[host, user, password, port, task_id],
        outputs=[dataset, connection_status]
    )
    
    view_btn.click(
        show_image,
        inputs=[dataset, index_input],
        outputs=[image_output, label_output, path_output]
    )

iface.launch()