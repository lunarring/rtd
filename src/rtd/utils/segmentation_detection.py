#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision
import time
import lunar_tools as lt
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

class HumanSeg:
    """
    This class is used for human segmentation in images. It uses a pre-trained model from PyTorch's model zoo 
    (deeplabv3_resnet101 by default) to predict the human body in the image and generate a mask. The mask can be 
    used for various applications like background removal, human pose estimation etc.
    
    Attributes:
        model: The pre-trained model used for human segmentation. available models: deeplabv3_resnet50 / deeplabv3_resnet101 / deeplabv3_mobilenet_v3_large
        size (tuple, optional): The desired size (height, width) for the output tensor. If provided, this overrides the downscaling_factor.
        resizing_factor (float, optional): The factor by which to downscale the input tensor. Defaults to None. Ignored if size is provided.
        preprocess: The preprocessing transformations applied on the input image before feeding it to the model.
        apply_smoothing (bool): Whether to smooth the segmentation mask with a Gaussian kernel (default: False).
        gaussian_kernel_size (int): Kernel size for Gaussian smoothing.
        gaussian_sigma (float): Sigma value for Gaussian smoothing.
    """
    def __init__(self, model_name='deeplabv3_resnet101', resizing_factor=None, size=None, device='cuda:0', apply_smoothing=False, gaussian_kernel_size=5, gaussian_sigma=1.0):
        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.sel_id = 15  # human body code
        self.resizing_factor = resizing_factor
        self.size = size
        self.mask = None
        self.apply_smoothing = apply_smoothing
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma
    
    def set_resizing_factor(self, resizing_factor):
        """
        This method sets the resizing factor for the HumanSeg class.

        Args:
            resizing_factor (float): The factor by which to downscale the input tensor.
        """
        self.resizing_factor = resizing_factor
    
    def get_mask(self, input_img):
        """
        This method generates a binary mask for the human body in the given image. The mask can be used for various applications like background removal, human pose estimation etc.

        Args:
            cam_img (np.ndarray): The input image in which the human body is to be segmented. The image should be in RGB format.

        Returns: float32
            Sets self.mask, a binary mask for the human body in the input image. The mask is of the same size as the input image.
        """
        if isinstance(input_img, np.ndarray) and input_img.dtype == np.float32:
            input_img = input_img.astype(np.uint8)
        
        if not isinstance(input_img, np.ndarray):
            input_img = np.array(input_img)
        
        orig_size = (input_img.shape[1], input_img.shape[0])
        input_tensor = self.preprocess(input_img.copy())
        
        if self.resizing_factor is not None or self.size is not None:
            if self.size is not None:
                print(self.size)
                input_tensor = lt.resize(input_tensor, size=self.size, resample_method='bilinear')
            else:
                input_tensor = lt.resize(input_tensor, resizing_factor=self.resizing_factor, resample_method='bilinear')
        
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(self.device)
        
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        
        output_predictions = output.argmax(0)
        if self.apply_smoothing:
            # Create a binary mask for the selected class and smooth it with a Gaussian kernel on the GPU
            mask_tensor = (output_predictions == self.sel_id).float().unsqueeze(0).unsqueeze(0)
            kernel_size = self.gaussian_kernel_size
            sigma = self.gaussian_sigma
            x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
            y = x.clone()
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            gaussian_kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            smoothed_mask = F.conv2d(mask_tensor, gaussian_kernel, padding=kernel_size//2)
            # smoothed_mask = (smoothed > 0.5).float()
            mask = smoothed_mask.squeeze().float().cpu().numpy()
            mask = lt.resize(mask*255, size=(orig_size[1], orig_size[0])) / 255
        else:
            mask = (output_predictions == self.sel_id).byte().cpu().numpy()
        
            if self.resizing_factor is not None or self.size is not None:
                mask = lt.resize(mask, size=(orig_size[1], orig_size[0]))
                mask = np.round(mask)
                mask = np.where(mask > 0.5, 1, 0)
        
        self.mask = mask
        
        return mask
    
    def apply_mask(self, input_img, mask_strength=1, invert_mask=False):
        """
        This method applies the previously generated mask to the input image.

        Args:
            input_img (np.ndarray): The input image to which the mask is to be applied. The image should be in RGB format.
            invert_mask (bool): If set to True, the mask will be inverted before being applied.

        Returns:
            np.ndarray: The masked image.
        """
        if self.mask is None:
            raise ValueError("No mask has been generated. Please call get_mask() first.")
        
        if isinstance(input_img, Image.Image):
            input_img = np.array(input_img)
        else:
            if input_img.dtype == np.float32:
                input_img = input_img.astype(np.uint8)
        
        assert input_img.shape[:2] == self.mask.shape, "The input image and the mask must have the same dimensions."
        assert 0 <= mask_strength <= 1, "mask_strength should be between 0 and 1"
        
        if self.apply_smoothing:
            input_img = input_img.astype(np.float32)

        mask_used = 1 - self.mask if invert_mask else self.mask
        masked_img = input_img * np.expand_dims(mask_used, axis=2)
        
        if mask_strength != 1:
            input_img = input_img.astype(float)
            masked_img = masked_img.astype(float)
            masked_img = (1 - mask_strength) * input_img + mask_strength * masked_img
            masked_img = np.round(masked_img)
            masked_img = np.clip(masked_img, 0, 255)
        
        final_img = masked_img.astype(np.uint8)
        return final_img


class FaceCropper:
    def __init__(self, model_path=None, padding=30, single_mode=True):
        if model_path is None:
            model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.yolo = YOLO(model_path)
        self.padding = int(padding)
        self.nmb_faces_present = 0
        self.single_mode = single_mode
        

    def set_padding(self, value):
        # print(f"set_padding called with {value}")
        if value < 0:
            raise ValueError("Padding must be non-negative")
        self.padding = int(value)

    def convert_input_img(self, input_img):
        if isinstance(input_img, torch.Tensor):
            input_img = input_img.cpu().numpy()
            input_img = np.clip(input_img * 255, 0, 255)
            
        if isinstance(input_img, Image.Image):
            input_img = np.array(input_img)

        if len(input_img.shape)==4:
            input_img = input_img[0, :, :, :]
            
        return input_img
        

    def get_cropping_coordinates(self, input_img):
        input_img = self.convert_input_img(input_img)
    
        # Run face detection
        try:
            face_results = self.yolo(input_img, verbose=False)
            self.nmb_faces_present = len(face_results[0].boxes.xyxy)
        except Exception as e:
            return None

        try:
            # If no face present, no need to do anything
            if len(face_results) == 0:
                return None
    
            # Initialize variables to store the maximum area and corresponding index
            cropping_coordinates = None
            list_cropping_coordinates = []
    
            if self.single_mode:
                max_area = 0
                # Loop through all detected faces and find largest one!
                for result in face_results:
                    # Retrieve the bounding box coordinates in the format (x1, y1, x2, y2)
                    x1, y1, x2, y2 = result.boxes.xyxy[0].cpu().numpy()
                    # Calculate the area of the bounding box
                    area = (x2 - x1) * (y2 - y1)
        
                    # Check if this area is the largest we've seen so far
                    if area > max_area:
                        max_area = area
                        # Update the cropping coordinates to the largest face found
                        cropping_coordinates = (x1, y1, x2, y2)
    
                if cropping_coordinates is None:
                    return None
                else:
                    list_cropping_coordinates.append(cropping_coordinates)

            else:
                # Take the first two faces and assign the left one to list_cropping_coordinates[0] and right one to list_cropping_coordinates[1]
                if self.nmb_faces_present >= 2:
                    result = face_results[0]
                    x1_first, y1_first, x2_first, y2_first = result.boxes.xyxy[0].cpu().numpy()
                    x1_second, y1_second, x2_second, y2_second = result.boxes.xyxy[1].cpu().numpy()

                    # Measure x positions to determine which is left and which is right
                    if x1_first < x1_second:
                        list_cropping_coordinates.append((x1_first, y1_first, x2_first, y2_first))
                        list_cropping_coordinates.append((x1_second, y1_second, x2_second, y2_second))
                    else:
                        list_cropping_coordinates.append((x1_second, y1_second, x2_second, y2_second))
                        list_cropping_coordinates.append((x1_first, y1_first, x2_first, y2_first))
                else:
                    return [None, None]

            list_cropping_coordinates_fixed = []
            for cropping_coordinates in list_cropping_coordinates:

                x1, y1, x2, y2 = cropping_coordinates
        
                # Adjust to square
                width = x2 - x1
                height = y2 - y1
        
                if width > height:
                    difference = (width - height) // 2
                    y1 = max(0, y1 - difference)
                    y2 = min(input_img.shape[0], y2 + difference)
                elif height > width:
                    difference = (height - width) // 2
                    x1 = max(0, x1 - difference)
                    x2 = min(input_img.shape[1], x2 + difference)
        
                if (x2 - x1) != (y2 - y1):
                    new_size = min(x2 - x1, y2 - y1)
                    x2 = x1 + new_size
                    y2 = y1 + new_size
        
                # Calculate new padding to center the padding around the box
                padding_x = (self.padding // 2, self.padding // 2)
                padding_y = (self.padding // 2, self.padding // 2)
        
                # Adjust padding if it causes the box to go out of image bounds
                if x1 - padding_x[0] < 0:
                    padding_x = (x1, self.padding - x1)
                if x2 + padding_x[1] > input_img.shape[1]:
                    padding_x = (self.padding - (input_img.shape[1] - x2), input_img.shape[1] - x2)
        
                if y1 - padding_y[0] < 0:
                    padding_y = (y1, self.padding - y1)
                if y2 + padding_y[1] > input_img.shape[0]:
                    padding_y = (self.padding - (input_img.shape[0] - y2), input_img.shape[0] - y2)
        
                # Apply adjusted padding
                x1 = max(0, x1 - padding_x[0])
                x2 = min(input_img.shape[1], x2 + padding_x[1])
                y1 = max(0, y1 - padding_y[0])
                y2 = min(input_img.shape[0], y2 + padding_y[1])
        
                cropping_coordinates_fixed = (int(x1), int(y1), int(x2), int(y2))
                list_cropping_coordinates_fixed.append(cropping_coordinates_fixed)

            if self.single_mode:
                return list_cropping_coordinates_fixed[0]
            else:
                return list_cropping_coordinates_fixed
            
        except Exception as e:
            # print(f"get_cropping_coordinates exception {e}")
            return None
        
    def apply_crop(self, input_img, cropping_coordinates):
        input_img = self.convert_input_img(input_img)
        if cropping_coordinates:
            return Image.fromarray(input_img).crop(cropping_coordinates)
        else:
            # print("warning! cropping coordinates are empty")
            return input_img





if __name__ == '__main__FaceCropper':
    import lunar_tools as lt
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    # # Load the image from the specified path
    # image_path = "/home/lugo/git/ComfyUI/input/example.png"
    # input_img = Image.open(image_path)
    # input_img = np.array(input_img)
    
    # input_img = np.array(input_img, dtype=np.float32) / 255.0
    # input_img = np.expand_dims(input_img, axis=0)
    
    
    # shape_cam=(1080, 1920) 
    # cam = lt.WebCam(shape_hw=shape_cam)

    face_cropper = FaceCropper(padding=30, single_mode=False)
    img_cam = Image.open('/home/lugo/git/tmp/img_twoface/img2.jpg')
    cropping_coordinates = face_cropper.get_cropping_coordinates(img_cam)
    face_cropper.apply_crop(img_cam, cropping_coordinates[0])
    

    #%%
    # cropping_coordinates = face_cropper.get_cropping_coordinates(img_cam)

    # if cropping_coordinates is not None:
    #     cam_img_cropped = Image.fromarray(img_cam).crop(cropping_coordinates)

    #     # Display the cropped image
    #     plt.imshow(cam_img_cropped)
    #     plt.ion()
    #     plt.show()
    # else:
    #     print("No face detected.")


if __name__ == '__main__':
    import lunar_tools as lt
    import time

    shape_cam = (576, 1024)
    cam = lt.WebCam(shape_hw=shape_cam)
    renderer = lt.Renderer(width=shape_cam[1], height=shape_cam[0], backend="pygame")

    human_seg = HumanSeg(resizing_factor=0.5, device='cuda:0')

    while True:
        start_time = time.time()

        cam_img = cam.get_img()
        cam_img = np.flip(cam_img, axis=1)

        human_seg.get_mask(cam_img)
        img = human_seg.apply_mask(cam_img)

        renderer.render(img)

        end_time = time.time()
        print(f"Iteration time: {end_time - start_time:.4f} seconds")