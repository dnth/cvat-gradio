# CVAT-Gradio

Programatically load CVAT bounding box datasets into Gradio using the CVAT SDK.

## Installation
Install the required packages using the following command:

```
pip install 'cvat_sdk[pytorch]' gradio
```

## Load Dataset into CVAT
Next, load your dataset into CVAT and annotate it. Optionally, you can load publicly available datasets.

In my setup, I loaded the [Aquarium dataset](https://public.roboflow.com/object-detection/aquarium) from Roboflow. The dataset consists of bounding boxe annotations for the following classes:

+ fish 
+ jellyfish 
+ penguins 
+ sharks 
+ puffins 
+ stingrays 
+ starfish

![cvat](./assets/cvat.png)

## Pulling Dataset from CVAT

We can use the CVAT SDK to pull the dataset from CVAT into a PyTorch `Dataset` object.

In the most basic form:

```python
from cvat_sdk import make_client

client = make_client(host, port=port, credentials=(user, password))

dataset = TaskVisionDataset(client, task_id)

## Visualize in Gradio and share
```

Read more [here](https://docs.cvat.ai/docs/api_sdk/sdk/pytorch-adapter/)


## Running the Gradio App

Finally, we can run the Gradio app using the following code snippet:

```python
python run_gradio.py
```
![gradio](./assets/gradio.png)