from rtd.dynamic_processor.base_dynamic_module import BaseDynamicClass
import numpy as np

class DynamicClass(BaseDynamicClass):
    def __init__(self):
        # Persistent circle state: a list of groups; each group is a list of circles.
        # Each circle is represented as a dict with keys 'x', 'y', and 'r'
        self.circles = []
        self.initialized = False
        self.rng = np.random.default_rng(42)

    def _init_circles(self, shape):
        H, W, _ = shape
        self.circles = []
        # Initialize 5 groups with 10 circles each.
        for i in range(5):
            group = []
            for j in range(10):
                x = self.rng.integers(0, W)
                y = self.rng.integers(0, H)
                r = self.rng.integers(5, 15)
                color = tuple(self.rng.integers(0,256, size=3).tolist())
                angle = self.rng.uniform(0, 2 * np.pi)
                vx = np.cos(angle)
                vy = np.sin(angle)
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
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        # Create coordinate grids for efficient circle drawing.
        Y, X = np.ogrid[:H, :W]
        for group in self.circles:
            for circle in group:
                cx = circle['x']
                cy = circle['y']
                r = circle['r']
                mask = (X - cx)**2 + (Y - cy)**2 <= r*r
                base_color = np.array(circle['color'], dtype=np.float32)
                noise = self.rng.integers(-20, 21, size=canvas[mask].shape)
                canvas[mask] = np.clip(base_color + noise, 0, 255)
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
        result = (img_camera * img_mask_segmentation) + circles_overlay
        result = np.clip(result, 0, 255)
        return result.astype(np.float32)
