import os
import cv2
import torch
import argparse
from tqdm import tqdm
from huggingface_hub import PyTorchModelHubMixin
import lunar_tools as lt
import time
import numpy as np

from ddcolor_model import DDColor
from infer import ImageColorizationPipeline

class ImageColorizationPipelineHF(ImageColorizationPipeline):
    # Nested merged model class combining DDColor with the HF-loading mixin
    class DDColor(DDColor, PyTorchModelHubMixin):
        pass

    def __init__(self, input_size=512):
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "piddnad/ddcolor_paper_tiny"
        ddcolor_model = self.DDColor.from_pretrained(model_name)
        self.model = ddcolor_model.to(self.device)
        self.model.eval()


if __name__ == "__main__":
    colorizer = ImageColorizationPipelineHF()

    # Setup webcam input using lunar_tools
    shape_cam = (576, 1024)
    cam = lt.WebCam(cam_id=-1, shape_hw=shape_cam)
    renderer = lt.Renderer(width=shape_cam[1], height=shape_cam[0], backend="pygame")

    while True:
        start_time = time.time()

        cam_img = cam.get_img()
        cam_img = np.flip(cam_img, axis=1)

        # Convert the webcam image from RGB to grayscale, but keep 3 channels
        gray = cv2.cvtColor(cam_img, cv2.COLOR_BGR2GRAY)
        gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        image_out = colorizer.process(gray_3)[:,:,::-1]

        renderer.render(image_out)

        end_time = time.time()
        print(f"Iteration time: {end_time - start_time:.4f} seconds")
