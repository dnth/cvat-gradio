import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from cvat_sdk import make_client
from cvat_sdk.pytorch import TaskVisionDataset, ExtractBoundingBoxes
from cvat_sdk.pytorch import Target
import colorsys

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def generate_distinct_colors(n):
    HSV_tuples = [(x * 1.0 / n, 0.8, 0.9) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return list(RGB_tuples) 


# Create a custom dataset class that inherits from TaskVisionDataset to get the image path from CVAT
class CustomTaskVision(TaskVisionDataset):
    def __getitem__(self, sample_index: int):
        sample = self._underlying.samples[sample_index]
        image_path = sample.frame_name
        sample_image = sample.media.load_image()
        sample_target = Target(sample.annotations, self._label_id_to_index)

        if self.transforms:
            sample_image, sample_target = self.transforms(sample_image, sample_target)
        return sample_image, sample_target, image_path


def plot_image_with_boxes(image, labels, index_to_name):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label_index in zip(labels["boxes"], labels["labels"]):

        # Get the class name from the index
        class_name = index_to_name.get(label_index.item(), f"Unknown ({label_index})")

        num_labels = len(index_to_name)
        colors = generate_distinct_colors(num_labels)
        color = colors[label_index % num_labels]

        x, y, x2, y2 = box.tolist()
        w, h = x2 - x, y2 - y
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        ax.text(x, y, class_name, color="white", backgroundcolor=color, fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def connect_and_load_dataset(host, user, password, port, task_id):
    def create_index_to_name_map(data):
        return {index: item["name"] for index, item in enumerate(data)}

    try:
        client = make_client(host, port=int(port), credentials=(user, password))
        dataset = CustomTaskVision(
            client,
            int(task_id),
            target_transform=ExtractBoundingBoxes(include_shape_types=["rectangle"]),
        )

        cvat_labels = client.tasks.retrieve(int(task_id)).get_labels()
        index_to_name = create_index_to_name_map(cvat_labels)

        return dataset, "Connected successfully", index_to_name, len(dataset)
    except Exception as e:
        return None, f"Connection failed: {str(e)}", None, 0


def show_image(dataset, index, index_to_name):
    if dataset is None:
        return None, "Please connect to CVAT first", ""
    if 0 <= index < len(dataset):
        image, label, img_path = dataset[index]
        fig = plot_image_with_boxes(image, label, index_to_name)
        label_str = (
            f"Boxes: {label['boxes'].tolist()}\n"
            f"Labels: {[index_to_name.get(l.item(), f'Unknown ({l})') for l in label['labels']]}"
        )
        return fig, label_str, img_path
    else:
        return None, "Invalid index", ""


def display_first_image(dataset, index_to_name):
    if dataset is None or len(dataset) == 0:
        return None, "No images available in the dataset", "", 0
    return show_image(dataset, 0, index_to_name) + (0,)

def view_next_image(dataset, current_index, index_to_name, dataset_length):
    if dataset is None:
        return None, "Please connect to CVAT first", "", current_index
    
    next_index = (current_index + 1) % dataset_length
    fig, label_str, img_path = show_image(dataset, next_index, index_to_name)
    return fig, label_str, img_path, next_index

def view_previous_image(dataset, current_index, index_to_name, dataset_length):
    if dataset is None:
        return None, "Please connect to CVAT first", "", current_index
    
    previous_index = (current_index - 1) % dataset_length
    fig, label_str, img_path = show_image(dataset, previous_index, index_to_name)
    return fig, label_str, img_path, previous_index

with gr.Blocks() as iface:
    gr.Markdown("# CVAT Dataset Viewer")
    gr.Markdown("Enter CVAT connection details and connect before viewing images.")

    with gr.Row():
        with gr.Column():
            with gr.Row():                
                host = gr.Textbox(label="CVAT Host", value="localhost")
                port = gr.Textbox(label="CVAT Port", value="8080")
                task_id = gr.Textbox(label="Task ID")
            user = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
        
            connect_btn = gr.Button("Connect")

        connection_status = gr.Textbox(label="Connection Status")

    with gr.Row():
        index_input = gr.Number(label="Image Index", precision=0)
        view_btn = gr.Button("View Image")
        view_previous_btn = gr.Button("View Previous")
        view_next_btn = gr.Button("View Next")

    image_output = gr.Plot()
    label_output = gr.Textbox(label="Labels", lines=5)
    path_output = gr.Text(label="Image Path")

    dataset = gr.State(None)
    index_to_name_state = gr.State(None)
    dataset_length = gr.State(0)

    connect_btn.click(
        connect_and_load_dataset,
        inputs=[host, user, password, port, task_id],
        outputs=[dataset, connection_status, index_to_name_state, dataset_length],
    ).then(
        display_first_image,
        inputs=[dataset, index_to_name_state],
        outputs=[image_output, label_output, path_output, index_input],
    )

    view_btn.click(
        show_image,
        inputs=[dataset, index_input, index_to_name_state],
        outputs=[image_output, label_output, path_output],
    )

    view_next_btn.click(
        view_next_image,
        inputs=[dataset, index_input, index_to_name_state, dataset_length],
        outputs=[image_output, label_output, path_output, index_input],
    )

    view_previous_btn.click(
        view_previous_image,
        inputs=[dataset, index_input, index_to_name_state, dataset_length],
        outputs=[image_output, label_output, path_output, index_input],
    )

iface.launch()