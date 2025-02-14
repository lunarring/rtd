from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import numpy as np
import torch
import math

class DynamicClass(BaseDynamicClass):
    def __init__(self):
        # Persistent circle state: a list of groups; each group is a list of circles.
        # Each circle is represented as a dict with keys 'x', 'y', and 'r'
        self.device = torch.device("cuda")
        self.circles = []
        self.initialized = False
        self.rng = torch.Generator(device=self.device).manual_seed(42)

    def _init_circles(self, shape):
        H, W, _ = shape
        self.circles = []
        # Initialize 5 groups with 10 circles each.
        for i in range(5):
            group = []
            for j in range(10):
                x = torch.randint(0, W, (1,), generator=self.rng, device=self.device).item()
                y = float(torch.randint(0, H, (1,), generator=self.rng, device=self.device).item())
                r = torch.randint(5, 15, (1,), generator=self.rng, device=self.device).item()
                color = tuple(torch.randint(0, 256, (3,), generator=self.rng, device=self.device).tolist())
                angle = torch.rand(1, generator=self.rng, device=self.device).item() * (2 * math.pi)
                vx = math.cos(angle)
                vy = math.sin(angle)
                group.append({'x': int(x), 'y': float(y), 'r': int(r), 'color': color, 'vx': float(vx), 'vy': float(vy)})
            self.circles.append(group)
        self.initialized = True

    def _update_circles(self, shape, speed_coef):
        H, W, _ = shape
        # Update each circle's position along its preset random direction.
        for group in self.circles:
            for circle in group:
                circle['x'] = (circle['x'] + speed_coef * 5 * circle['vx']) % W
                circle['y'] = (circle['y'] + speed_coef * 5 * circle['vy']) % H

    def _draw_circles(self, shape):
        H, W, _ = shape
        canvas = torch.zeros((H, W, 3), dtype=torch.float32, device=self.device)
        Y = torch.arange(H, device=self.device).view(H, 1)
        X = torch.arange(W, device=self.device).view(1, W)
        for group in self.circles:
            for circle in group:
                cx = circle['x']
                cy = circle['y']
                r = circle['r']
                mask = (X - cx)**2 + (Y - cy)**2 <= r*r
                base_color = torch.tensor(circle['color'], dtype=torch.float32, device=self.device)
                noise = torch.randint(-20, 21, canvas[mask].shape, generator=self.rng, device=self.device, dtype=torch.int32).to(torch.float32)
                canvas[mask] = torch.clamp(base_color + noise, 0, 255)
        return canvas

    def process(self, img_camera: np.ndarray,
                img_mask_segmentation: np.ndarray,
                img_diffusion: np.ndarray,
                dynamic_func_coef: float) -> np.ndarray:
        if not self.initialized:
            self._init_circles(img_camera.shape)
        self._update_circles(img_camera.shape, dynamic_func_coef)
        circles_overlay = self._draw_circles(img_camera.shape)
        # Apply segmentation mask to camera image and combine with circles overlay
        img_camera_t = torch.tensor(img_camera, dtype=torch.float32, device=self.device)
        img_mask_t = torch.tensor(img_mask_segmentation, dtype=torch.float32, device=self.device)
        result = (img_camera_t * img_mask_t) + circles_overlay
        result = torch.clamp(result, 0, 255)
        return result.cpu().numpy()
