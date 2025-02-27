import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image


class NavierStokesSimulator:
    """
    A class for simulating fluid dynamics using the Navier-Stokes equations.
    This implementation is based on Jos Stam's "Stable Fluids" algorithm.
    
    Attributes:
        width (int): Width of the simulation grid.
        height (int): Height of the simulation grid.
        device (torch.device): Device to run the simulation on.
        density (torch.Tensor): Fluid density field.
        velocity_x (torch.Tensor): Velocity field in x direction.
        velocity_y (torch.Tensor): Velocity field in y direction.
        viscosity (float): Fluid viscosity coefficient.
        diffusion (float): Fluid diffusion coefficient.
        dt (float): Time step for simulation.
    """
    
    def __init__(self, width, height, device="cuda:0", viscosity=0.0001, diffusion=0.0, dt=0.2):
        """
        Initialize the Navier-Stokes simulator.
        
        Args:
            width (int): Width of the simulation grid.
            height (int): Height of the simulation grid.
            device (str): Device to run the simulation on.
            viscosity (float): Fluid viscosity coefficient.
            diffusion (float): Fluid diffusion coefficient.
            dt (float): Time step for simulation.
        """
        self.width = width
        self.height = height
        self.device = torch.device(device)
        self.viscosity = viscosity
        self.diffusion = diffusion
        self.dt = dt
        
        # Initialize fields
        self.density = torch.zeros((height, width), device=self.device)
        self.velocity_x = torch.zeros((height, width), device=self.device)
        self.velocity_y = torch.zeros((height, width), device=self.device)
        
        # Previous state
        self.prev_density = torch.zeros((height, width), device=self.device)
        self.prev_velocity_x = torch.zeros((height, width), device=self.device)
        self.prev_velocity_y = torch.zeros((height, width), device=self.device)
        
        # For visualization
        self.dye = torch.zeros((height, width, 3), device=self.device)
        self.prev_dye = torch.zeros((height, width, 3), device=self.device)
    
    def add_density(self, x, y, amount):
        """
        Add density at a specific position.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            amount (float): Amount of density to add.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.density[y, x] += amount
    
    def add_velocity(self, x, y, amount_x, amount_y):
        """
        Add velocity at a specific position.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            amount_x (float): Amount of velocity to add in x direction.
            amount_y (float): Amount of velocity to add in y direction.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.velocity_x[y, x] += amount_x
            self.velocity_y[y, x] += amount_y
    
    def add_dye(self, x, y, color, radius=5):
        """
        Add colored dye at a specific position.
        
        Args:
            x (int): X coordinate.
            y (int): Y coordinate.
            color (list): RGB color values [r, g, b] in range [0, 1].
            radius (int): Radius of the dye spot.
        """
        # Create a grid of coordinates
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        
        # Calculate distance from (x, y)
        distance = torch.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        
        # Create a mask for points within the radius
        mask = distance < radius
        
        # Add dye color where mask is True
        color_tensor = torch.tensor(color, device=self.device)
        for c in range(3):
            self.dye[..., c] = torch.where(mask, self.dye[..., c] + color_tensor[c], self.dye[..., c])
        
        # Clamp values to [0, 1]
        self.dye = torch.clamp(self.dye, 0, 1)
    
    def diffuse(self, field, prev_field, diffusion, dt):
        """
        Diffuse a field using Gauss-Seidel relaxation.
        
        Args:
            field (torch.Tensor): Current field.
            prev_field (torch.Tensor): Previous field.
            diffusion (float): Diffusion coefficient.
            dt (float): Time step.
            
        Returns:
            torch.Tensor: Diffused field.
        """
        a = dt * diffusion * self.width * self.height
        
        # Copy prev_field to field initially
        field.copy_(prev_field)
        
        # Gauss-Seidel relaxation
        for _ in range(20):
            field[1:-1, 1:-1] = (
                prev_field[1:-1, 1:-1] + 
                a * (
                    field[0:-2, 1:-1] + 
                    field[2:, 1:-1] + 
                    field[1:-1, 0:-2] + 
                    field[1:-1, 2:]
                )
            ) / (1 + 4 * a)
            
            # Set boundary conditions
            self.set_boundary(field)
        
        return field
    
    def advect(self, field, prev_field, velocity_x, velocity_y, dt):
        """
        Advect a field using semi-Lagrangian advection.
        
        Args:
            field (torch.Tensor): Current field.
            prev_field (torch.Tensor): Previous field.
            velocity_x (torch.Tensor): Velocity field in x direction.
            velocity_y (torch.Tensor): Velocity field in y direction.
            dt (float): Time step.
            
        Returns:
            torch.Tensor: Advected field.
        """
        # Create grid coordinates
        y_indices, x_indices = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing='ij'
        )
        
        # Calculate backtraced positions
        x_back = x_indices - dt * velocity_x * self.width
        y_back = y_indices - dt * velocity_y * self.height
        
        # Clamp to grid boundaries
        x_back = torch.clamp(x_back, 0, self.width - 1.001)
        y_back = torch.clamp(y_back, 0, self.height - 1.001)
        
        # Get integer and fractional parts
        x0 = torch.floor(x_back).long()
        y0 = torch.floor(y_back).long()
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Ensure indices are within bounds
        x1 = torch.clamp(x1, 0, self.width - 1)
        y1 = torch.clamp(y1, 0, self.height - 1)
        
        # Get fractional parts
        sx = x_back - x0.float()
        sy = y_back - y0.float()
        
        # Bilinear interpolation
        field = (
            prev_field[y0, x0] * (1 - sx) * (1 - sy) +
            prev_field[y0, x1] * sx * (1 - sy) +
            prev_field[y1, x0] * (1 - sx) * sy +
            prev_field[y1, x1] * sx * sy
        )
        
        # Set boundary conditions
        self.set_boundary(field)
        
        return field
    
    def project(self, velocity_x, velocity_y):
        """
        Project the velocity field to be mass-conserving (divergence-free).
        
        Args:
            velocity_x (torch.Tensor): Velocity field in x direction.
            velocity_y (torch.Tensor): Velocity field in y direction.
            
        Returns:
            tuple: Updated velocity fields (velocity_x, velocity_y).
        """
        # Calculate divergence
        divergence = torch.zeros((self.height, self.width), device=self.device)
        pressure = torch.zeros((self.height, self.width), device=self.device)
        
        divergence[1:-1, 1:-1] = -0.5 * (
            velocity_x[1:-1, 2:] - velocity_x[1:-1, 0:-2] +
            velocity_y[2:, 1:-1] - velocity_y[0:-2, 1:-1]
        ) / self.width
        
        self.set_boundary(divergence)
        
        # Solve Poisson equation
        for _ in range(20):
            pressure[1:-1, 1:-1] = (
                divergence[1:-1, 1:-1] + 
                pressure[0:-2, 1:-1] + 
                pressure[2:, 1:-1] + 
                pressure[1:-1, 0:-2] + 
                pressure[1:-1, 2:]
            ) / 4
            
            self.set_boundary(pressure)
        
        # Subtract pressure gradient from velocity
        velocity_x[1:-1, 1:-1] -= 0.5 * (pressure[1:-1, 2:] - pressure[1:-1, 0:-2]) * self.width
        velocity_y[1:-1, 1:-1] -= 0.5 * (pressure[2:, 1:-1] - pressure[0:-2, 1:-1]) * self.height
        
        self.set_boundary(velocity_x)
        self.set_boundary(velocity_y)
        
        return velocity_x, velocity_y
    
    def set_boundary(self, field):
        """
        Set boundary conditions for a field.
        
        Args:
            field (torch.Tensor): Field to set boundary conditions for.
            
        Returns:
            torch.Tensor: Field with boundary conditions set.
        """
        # Set edges
        field[0, :] = field[1, :]  # Top
        field[-1, :] = field[-2, :]  # Bottom
        field[:, 0] = field[:, 1]  # Left
        field[:, -1] = field[:, -2]  # Right
        
        # Set corners
        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])  # Top-left
        field[0, -1] = 0.5 * (field[1, -1] + field[0, -2])  # Top-right
        field[-1, 0] = 0.5 * (field[-2, 0] + field[-1, 1])  # Bottom-left
        field[-1, -1] = 0.5 * (field[-2, -1] + field[-1, -2])  # Bottom-right
        
        return field
    
    def step(self):
        """
        Perform one step of the fluid simulation.
        """
        # Swap buffers
        self.velocity_x, self.prev_velocity_x = self.prev_velocity_x, self.velocity_x
        self.velocity_y, self.prev_velocity_y = self.prev_velocity_y, self.velocity_y
        self.density, self.prev_density = self.prev_density, self.density
        self.dye, self.prev_dye = self.prev_dye, self.dye
        
        # Diffuse velocity
        self.velocity_x = self.diffuse(self.velocity_x, self.prev_velocity_x, self.viscosity, self.dt)
        self.velocity_y = self.diffuse(self.velocity_y, self.prev_velocity_y, self.viscosity, self.dt)
        
        # Project to make velocity field mass-conserving
        self.velocity_x, self.velocity_y = self.project(self.velocity_x, self.velocity_y)
        
        # Advect velocity
        self.prev_velocity_x, self.prev_velocity_y = self.velocity_x.clone(), self.velocity_y.clone()
        self.velocity_x = self.advect(self.velocity_x, self.prev_velocity_x, self.prev_velocity_x, self.prev_velocity_y, self.dt)
        self.velocity_y = self.advect(self.velocity_y, self.prev_velocity_y, self.prev_velocity_x, self.prev_velocity_y, self.dt)
        
        # Project again
        self.velocity_x, self.velocity_y = self.project(self.velocity_x, self.velocity_y)
        
        # Diffuse density
        self.density = self.diffuse(self.density, self.prev_density, self.diffusion, self.dt)
        
        # Advect density
        self.density = self.advect(self.density, self.prev_density, self.velocity_x, self.velocity_y, self.dt)
        
        # Advect dye for visualization
        for c in range(3):
            self.dye[..., c] = self.advect(
                self.dye[..., c], 
                self.prev_dye[..., c], 
                self.velocity_x, 
                self.velocity_y, 
                self.dt
            )
    
    def get_flow_field(self):
        """
        Get the current flow field as a tensor suitable for image warping.
        
        Returns:
            torch.Tensor: Flow field tensor of shape (H, W, 2).
        """
        # Scale the velocity to pixel displacements
        flow = torch.stack([self.velocity_x, self.velocity_y], dim=2)
        return flow
    
    def get_density_visualization(self):
        """
        Get a visualization of the density field.
        
        Returns:
            torch.Tensor: RGB visualization tensor of shape (H, W, 3).
        """
        try:
            # Normalize density to [0, 1]
            max_density = torch.max(self.density)
            if max_density < 1e-8:
                # If density is too low, create a visible but dim visualization
                normalized = self.density * 0.1
            else:
                normalized = self.density / (max_density + 1e-8)
            
            # Ensure values are in valid range
            normalized = torch.clamp(normalized, 0.0, 1.0)
            
            # Create RGB visualization
            rgb = torch.stack([normalized, normalized, normalized], dim=2)
            
            # Add a minimum brightness to make it visible
            rgb = torch.clamp(rgb + 0.05, 0.0, 1.0)
            
            return rgb
        except Exception as e:
            print(f"Error in density visualization: {e}")
            # Return a fallback visualization
            return torch.ones((self.height, self.width, 3), device=self.device) * 0.2
    
    def get_dye_visualization(self):
        """
        Get a visualization of the dye field.
        
        Returns:
            torch.Tensor: RGB visualization tensor of shape (H, W, 3).
        """
        return self.dye
    
    def get_velocity_visualization(self):
        """
        Get a visualization of the velocity field.
        
        Returns:
            torch.Tensor: RGB visualization tensor of shape (H, W, 3).
        """
        try:
            # Calculate velocity magnitude
            magnitude = torch.sqrt(self.velocity_x**2 + self.velocity_y**2)
            max_magnitude = torch.max(magnitude)
            
            if max_magnitude < 1e-8:
                # If velocity is too low, create a visible but dim visualization
                normalized_magnitude = magnitude * 0.1
                # Return a simple visualization for low velocity
                rgb = torch.ones((self.height, self.width, 3), device=self.device) * 0.2
                return rgb
            
            normalized_magnitude = magnitude / (max_magnitude + 1e-8)
            
            # Calculate velocity direction (angle)
            angle = torch.atan2(self.velocity_y, self.velocity_x) / (2 * np.pi) + 0.5  # Normalize to [0, 1]
            
            # Create RGB visualization tensor
            rgb = torch.zeros((self.height, self.width, 3), device=self.device)
            
            # Use vectorized operations instead of loops for better performance
            # Convert HSV to RGB (vectorized version)
            h = angle * 6  # Hue * 6
            c = normalized_magnitude  # Chroma
            x = c * (1 - torch.abs((h % 2) - 1))  # X component in HSV to RGB conversion
            
            # Create masks for different hue regions
            mask0 = (h < 1)
            mask1 = (h >= 1) & (h < 2)
            mask2 = (h >= 2) & (h < 3)
            mask3 = (h >= 3) & (h < 4)
            mask4 = (h >= 4) & (h < 5)
            mask5 = (h >= 5)
            
            # Apply masks to create RGB values
            rgb[..., 0] = c * mask0 + x * mask1 + 0 * mask2 + 0 * mask3 + x * mask4 + c * mask5
            rgb[..., 1] = x * mask0 + c * mask1 + c * mask2 + x * mask3 + 0 * mask4 + 0 * mask5
            rgb[..., 2] = 0 * mask0 + 0 * mask1 + x * mask2 + c * mask3 + c * mask4 + x * mask5
            
            # Scale by magnitude to make high-velocity areas brighter
            rgb = rgb * normalized_magnitude.unsqueeze(-1)
            
            # Add a minimum brightness to make it visible
            rgb = torch.clamp(rgb + 0.1, 0.0, 1.0)
            
            return rgb
        except Exception as e:
            print(f"Error in velocity visualization: {e}")
            # Return a fallback visualization
            return torch.ones((self.height, self.width, 3), device=self.device) * 0.2
    
    def load_image(self, image_path, target_size=None):
        """
        Load an image from disk and convert it to a tensor.
        
        Args:
            image_path (str): Path to the image file.
            target_size (tuple, optional): Target size (width, height) to resize the image to.
                                          If None, the image is not resized.
        
        Returns:
            torch.Tensor: Image tensor of shape (H, W, C).
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize if target_size is provided
        if target_size is not None:
            img = img.resize(target_size)
        
        # Convert to numpy array and then to tensor
        img_np = np.array(img) / 255.0
        img_tensor = torch.tensor(img_np, dtype=torch.float32, device=self.device)
        
        return img_tensor
    
    def warp_tensor(self, img, flow):
        """
        Warps a given image tensor using the provided optical flow.
        Assumes:
          - img is a torch.Tensor of shape (H, W, C)
          - flow is a torch.Tensor with shape (H, W, 2) in pixel displacements.
        """
        H, W = img.shape[:2]
        # Rearrange image to (1, C, H, W) for grid_sample
        img_batch = img.permute(2, 0, 1).unsqueeze(0)
        
        # Create a normalized coordinate grid in the range [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=self.device),
            torch.linspace(-1, 1, W, device=self.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=2)  # shape (H, W, 2)
        
        # Convert flow from pixel displacement to normalized coordinates.
        flow_norm_x = flow[..., 0] * (2.0 / (W - 1))
        flow_norm_y = flow[..., 1] * (2.0 / (H - 1))
        flow_norm = torch.stack((flow_norm_x, flow_norm_y), dim=2)
        
        # Compute the new grid positions by adding the normalized flow
        grid_warp = grid + flow_norm
        grid_warp = grid_warp.unsqueeze(0)  # add batch dimension
        
        # Warp the image using grid_sample.
        warped_img = F.grid_sample(img_batch, grid_warp, mode='bilinear', padding_mode='border', align_corners=True)
        # Return to original shape (H, W, C)
        warped_img = warped_img.squeeze(0).permute(1, 2, 0)
        return warped_img
    
    def visualize_with_renderer(self, renderer, visualization_type='dye', blend_factor=1.0, background_image=None, resize_to=None):
        """
        Visualize the fluid simulation using the provided renderer.
        
        Args:
            renderer: An instance of lt.Renderer to render the visualization.
            visualization_type (str): Type of visualization to render. Options:
                - 'dye': Colored dye visualization
                - 'density': Density field visualization
                - 'velocity': Velocity field visualization
                - 'combined': Blend of dye and velocity visualizations
            blend_factor (float): Factor for blending visualizations in 'combined' mode.
            background_image (numpy.ndarray, optional): Background image to blend with the visualization.
            resize_to (tuple, optional): Target size (width, height) to resize the visualization to.
                                         If None, the original simulation size is used.
        """
        # Get the appropriate visualization based on the type
        if visualization_type == 'dye':
            vis_tensor = self.get_dye_visualization()
        elif visualization_type == 'density':
            vis_tensor = self.get_density_visualization()
        elif visualization_type == 'velocity':
            vis_tensor = self.get_velocity_visualization()
        elif visualization_type == 'combined':
            dye_vis = self.get_dye_visualization()
            vel_vis = self.get_velocity_visualization()
            vis_tensor = dye_vis * blend_factor + vel_vis * (1 - blend_factor)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        # Convert tensor to numpy array for rendering
        vis_np = vis_tensor.cpu().numpy()
        
        # Scale to 0-255 range for rendering
        vis_np = (vis_np * 255).astype(np.uint8)
        
        # Resize if needed
        if resize_to is not None:
            vis_np = cv2.resize(vis_np, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_LINEAR)
        
        # Blend with background image if provided
        if background_image is not None:
            # Ensure background image has the same shape as the visualization
            if background_image.shape[:2] != vis_np.shape[:2]:
                background_image = cv2.resize(background_image, (vis_np.shape[1], vis_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Blend the visualization with the background
            alpha = 0.7  # Blend factor
            vis_np = cv2.addWeighted(vis_np, alpha, background_image, 1 - alpha, 0)
        
        # Render the visualization
        renderer.render(vis_np)
        
        return vis_np
    
    @staticmethod
    def run_example(width=800, height=600, fullscreen=False, device="cuda:0"):
        """
        Run an example fluid simulation visualization using lt.Renderer.
        
        Args:
            width (int): Width of the simulation and rendering window.
            height (int): Height of the simulation and rendering window.
            fullscreen (bool): Whether to run in fullscreen mode.
            device (str): Device to run the simulation on.
        """
        try:
            import lunar_tools as lt
            import time
            import random
            import numpy as np
            import cv2
            import pygame
            import os
            
            # Initialize pygame for key handling and display
            pygame.init()
            print("Pygame initialized successfully")
            
            # Create a visible pygame window for key handling - make it larger and more prominent
            pygame_window = pygame.display.set_mode((600, 400))
            pygame.display.set_caption("NAVIER-STOKES CONTROL PANEL - CLICK HERE TO USE KEYBOARD")
            print("Pygame window created")
            
            # Initialize the simulator
            print(f"Initializing simulator with device={device}, width={width}, height={height}")
            simulator = NavierStokesSimulator(
                width=width,
                height=height,
                device=device,
                viscosity=0.0001,
                diffusion=0.0,
                dt=0.1
            )
            print("Simulator initialized successfully")
            
            # Choose rendering method based on availability
            use_lunar_tools = True
            use_opencv = True
            
            try:
                # Initialize the renderer with lunar_tools
                print("Attempting to initialize lunar_tools renderer...")
                renderer = lt.Renderer(
                    width=width,
                    height=height,
                    backend="gl",
                    do_fullscreen=fullscreen
                )
                print("Lunar tools renderer initialized successfully")
                
                # Initialize FPS tracking if available
                try:
                    fps_tracker = lt.FPSTracker()
                    has_fps_tracker = True
                    print("FPS tracker initialized")
                except Exception as fps_error:
                    print(f"FPS tracker not available: {fps_error}")
                    has_fps_tracker = False
                    
            except Exception as e:
                print(f"Lunar Tools renderer not available: {e}")
                print("Falling back to OpenCV renderer only")
                use_lunar_tools = False
                has_fps_tracker = False
            
            # Always create OpenCV window as a backup
            print("Creating OpenCV window...")
            cv2.namedWindow("Navier-Stokes Fluid Simulation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Navier-Stokes Fluid Simulation", width, height)
            print("OpenCV window created")
            
            # Main simulation loop
            running = True
            last_dye_time = time.time()
            dye_interval = 0.05  # Add dye more frequently (every 0.05 seconds)
            
            # Default visualization settings
            vis_type = 'dye'  # Start with dye visualization as it's most visible
            blend_factor = 0.7
            
            # Mouse interaction variables
            mouse_x, mouse_y = width // 2, height // 2  # Default mouse position
            prev_mouse_x, prev_mouse_y = mouse_x, mouse_y  # Previous mouse position
            mouse_left_clicked = False
            mouse_right_clicked = False
            
            # Mouse callback function for OpenCV window
            def mouse_callback(event, x, y, flags, param):
                nonlocal mouse_x, mouse_y, mouse_left_clicked, mouse_right_clicked, prev_mouse_x, prev_mouse_y
                
                # Update previous position before setting new position
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                mouse_x, mouse_y = x, y
                
                # Check for mouse clicks
                if event == cv2.EVENT_LBUTTONDOWN:
                    mouse_left_clicked = True
                elif event == cv2.EVENT_RBUTTONDOWN:
                    mouse_right_clicked = True
            
            # Set the mouse callback for the OpenCV window
            cv2.setMouseCallback("Navier-Stokes Fluid Simulation", mouse_callback)
            print("Mouse interaction enabled - click anywhere to add velocity")
            
            # Print controls
            print("\n=== CONTROLS ===")
            print("  Mouse: Left-click to add velocity, Right-click to add colored dye")
            print("  1: Dye visualization (colored fluid)")
            print("  2: Density visualization (grayscale fluid density)")
            print("  3: Velocity visualization (color-coded velocity field)")
            print("  4: Combined visualization (blend of dye and velocity)")
            print("  +/-: Increase/decrease blend factor for combined visualization")
            print("  V/v: Increase/decrease viscosity (fluid thickness)")
            print("  D/d: Increase/decrease diffusion (how quickly dye spreads)")
            print("  Space: Add velocity burst at center without adding dye")
            print("  ESC: Quit")
            print("================\n")
            
            print("IMPORTANT: Click on the 'NAVIER-STOKES CONTROL PANEL' window to use keyboard controls!")
            print("If keyboard controls don't work in the pygame window, try clicking on the OpenCV window and using keyboard there.")
            
            # Add initial dye and velocity to make visualization immediately visible
            print("Loading ice.jpg image as initial pattern...")
            
            try:
                # Load the ice image
                ice_img_path = os.path.join("materials", "images", "ice.jpg")
                if not os.path.exists(ice_img_path):
                    print(f"Warning: {ice_img_path} not found. Creating a fallback pattern.")
                    # Add a central burst as fallback
                    center_x, center_y = width // 2, height // 2
                    simulator.add_dye(center_x, center_y, [1.0, 0.0, 0.0], radius=80)  # Red
                    print("Added central red dye burst as fallback")
                else:
                    # Load the image and use it as initial dye
                    ice_img = simulator.load_image(ice_img_path, target_size=(width, height))
                    simulator.dye = ice_img.clone()
                    print(f"Successfully loaded {ice_img_path} as initial pattern")
            except Exception as e:
                print(f"Error loading ice image: {e}")
                # Add a central burst as fallback
                center_x, center_y = width // 2, height // 2
                simulator.add_dye(center_x, center_y, [1.0, 0.0, 0.0], radius=80)  # Red
                print("Added central red dye burst as fallback due to error")
            
            # Add outward velocity from center
            center_x, center_y = width // 2, height // 2
            for i in range(16):
                angle = i * np.pi / 8
                vel_x = 10 * np.cos(angle)
                vel_y = 10 * np.sin(angle)
                simulator.add_velocity(center_x, center_y, vel_x, vel_y)
            print("Added outward velocity from center")
            
            # Run a few simulation steps to get initial flow
            print("Running initial simulation steps...")
            for _ in range(10):
                simulator.step()
            print("Initial simulation steps completed")
            
            print("\nVisualization started!")
            print("Current mode: DYE visualization")
            print("If you don't see anything, check if the window is behind other windows.")
            print("Focus on the pygame control window to use keyboard controls.")
            
            clock = pygame.time.Clock()
            frame_count = 0
            last_key_time = 0
            key_feedback = None
            
            # Function to handle key presses (used for both pygame and OpenCV)
            def handle_key(key):
                nonlocal vis_type, blend_factor, key_feedback, last_key_time
                
                # Set feedback message and time
                last_key_time = time.time()
                
                if key == '1':
                    vis_type = 'dye'
                    key_feedback = "SWITCHED TO DYE VISUALIZATION"
                    print("\n=== SWITCHED TO DYE VISUALIZATION ===")
                    print("Shows the colored dye as it flows through the fluid")
                    print("This is the most visually appealing visualization")
                elif key == '2':
                    vis_type = 'density'
                    key_feedback = "SWITCHED TO DENSITY VISUALIZATION"
                    print("\n=== SWITCHED TO DENSITY VISUALIZATION ===")
                    print("Shows the density of the fluid as grayscale")
                    print("Brighter areas indicate higher fluid density")
                elif key == '3':
                    vis_type = 'velocity'
                    key_feedback = "SWITCHED TO VELOCITY VISUALIZATION"
                    print("\n=== SWITCHED TO VELOCITY VISUALIZATION ===")
                    print("Shows the velocity field of the fluid")
                    print("Color indicates direction, brightness indicates speed")
                    print("Red/yellow: rightward flow, Green: downward flow")
                    print("Cyan/Blue: leftward flow, Magenta: upward flow")
                elif key == '4':
                    vis_type = 'combined'
                    key_feedback = "SWITCHED TO COMBINED VISUALIZATION"
                    print("\n=== SWITCHED TO COMBINED VISUALIZATION ===")
                    print(f"Blend of dye and velocity (blend factor: {blend_factor:.2f})")
                    print("Use +/- keys to adjust the blend factor")
                elif key == '+':
                    blend_factor = min(1.0, blend_factor + 0.05)
                    key_feedback = f"BLEND FACTOR: {blend_factor:.2f}"
                    print(f"Blend factor increased to: {blend_factor:.2f}")
                    print("Higher values show more dye, lower values show more velocity")
                elif key == '-':
                    blend_factor = max(0.0, blend_factor - 0.05)
                    key_feedback = f"BLEND FACTOR: {blend_factor:.2f}"
                    print(f"Blend factor decreased to: {blend_factor:.2f}")
                    print("Higher values show more dye, lower values show more velocity")
                elif key.lower() == 'v':
                    if key == 'V':  # uppercase V
                        simulator.viscosity = min(0.001, simulator.viscosity * 1.2)
                        key_feedback = f"VISCOSITY INCREASED: {simulator.viscosity:.6f}"
                        print(f"Viscosity increased to: {simulator.viscosity:.6f}")
                        print("Higher viscosity makes the fluid thicker and flow more slowly")
                    else:  # lowercase v
                        simulator.viscosity = max(0.00001, simulator.viscosity * 0.8)
                        key_feedback = f"VISCOSITY DECREASED: {simulator.viscosity:.6f}"
                        print(f"Viscosity decreased to: {simulator.viscosity:.6f}")
                        print("Lower viscosity makes the fluid thinner and flow more quickly")
                elif key.lower() == 'd':
                    if key == 'D':  # uppercase D
                        simulator.diffusion = min(0.001, simulator.diffusion + 0.0001)
                        key_feedback = f"DIFFUSION INCREASED: {simulator.diffusion:.6f}"
                        print(f"Diffusion increased to: {simulator.diffusion:.6f}")
                        print("Higher diffusion makes the dye spread out more quickly")
                    else:  # lowercase d
                        simulator.diffusion = max(0.0, simulator.diffusion - 0.0001)
                        key_feedback = f"DIFFUSION DECREASED: {simulator.diffusion:.6f}"
                        print(f"Diffusion decreased to: {simulator.diffusion:.6f}")
                        print("Lower diffusion makes the dye spread out more slowly")
                elif key == ' ':  # space
                    # Add velocity at center without adding dye
                    x = width // 2
                    y = height // 2
                    # color = [random.random(), random.random(), random.random()]
                    # simulator.add_dye(x, y, color, radius=50)
                    # Add outward velocity
                    for i in range(8):
                        angle = i * np.pi / 4
                        vel_x = 5 * np.cos(angle)
                        vel_y = 5 * np.sin(angle)
                        simulator.add_velocity(x, y, vel_x, vel_y)
                    key_feedback = "ADDED VELOCITY BURST"
                    print("Added velocity burst at center without adding dye")
                elif key == 'escape':
                    return False  # Signal to quit
                
                return True  # Continue running
            
            while running:
                # Process pygame events for key handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        print("Quit event received")
                    elif event.type == pygame.KEYDOWN:
                        # Handle keyboard controls
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            print("ESC pressed - exiting")
                        elif event.key == pygame.K_1:
                            handle_key('1')
                        elif event.key == pygame.K_2:
                            handle_key('2')
                        elif event.key == pygame.K_3:
                            handle_key('3')
                        elif event.key == pygame.K_4:
                            handle_key('4')
                        elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                            handle_key('+')
                        elif event.key == pygame.K_MINUS:
                            handle_key('-')
                        elif event.key == pygame.K_v:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                handle_key('V')
                            else:
                                handle_key('v')
                        elif event.key == pygame.K_d:
                            if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                                handle_key('D')
                            else:
                                handle_key('d')
                        elif event.key == pygame.K_SPACE:
                            handle_key(' ')
                
                # Handle mouse interaction for velocity (left click)
                if mouse_left_clicked:
                    # Calculate velocity direction from mouse movement
                    dx = mouse_x - prev_mouse_x
                    dy = mouse_y - prev_mouse_y
                    
                    # If there's no movement, use a small default velocity
                    if abs(dx) < 1 and abs(dy) < 1:
                        dx, dy = 1, 0  # Default rightward velocity
                    
                    # Normalize the direction
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx /= length
                        dy /= length
                    
                    # Add velocity at mouse position
                    velocity_strength = 20.0  # Adjust this value to control velocity strength
                    
                    # Add velocity in a small area around the mouse position
                    radius = 30  # Radius around click point to add velocity
                    for r in range(radius):
                        for theta in range(8):  # 8 directions around the point
                            angle = theta * np.pi / 4
                            x = int(mouse_x + r * np.cos(angle))
                            y = int(mouse_y + r * np.sin(angle))
                            if 0 <= x < width and 0 <= y < height:
                                # Decrease strength as we move away from center
                                factor = 1.0 - (r / radius)
                                simulator.add_velocity(x, y, dx * velocity_strength * factor, dy * velocity_strength * factor)
                    
                    print(f"Added velocity at ({mouse_x}, {mouse_y}) with direction ({dx:.2f}, {dy:.2f})")
                    key_feedback = f"ADDED VELOCITY AT ({mouse_x}, {mouse_y})"
                    last_key_time = time.time()
                    
                    # Reset the click flag
                    mouse_left_clicked = False
                
                # Handle mouse interaction for dye (right click)
                if mouse_right_clicked:
                    # Generate a random bright color
                    color = [random.random(), random.random(), random.random()]
                    # Ensure the color is bright enough
                    while sum(color) < 1.5:  # Adjust this threshold for desired brightness
                        color = [random.random(), random.random(), random.random()]
                    
                    # Add dye at mouse position
                    simulator.add_dye(mouse_x, mouse_y, color, radius=40)
                    
                    print(f"Added dye at ({mouse_x}, {mouse_y}) with color ({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
                    key_feedback = f"ADDED DYE AT ({mouse_x}, {mouse_y})"
                    last_key_time = time.time()
                    
                    # Reset the click flag
                    mouse_right_clicked = False
                
                # Also check for OpenCV key presses as a backup
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # If a key was pressed
                    if key == 27:  # ESC
                        running = handle_key('escape')
                    elif key == ord('1'):
                        handle_key('1')
                    elif key == ord('2'):
                        handle_key('2')
                    elif key == ord('3'):
                        handle_key('3')
                    elif key == ord('4'):
                        handle_key('4')
                    elif key == ord('+') or key == ord('='):
                        handle_key('+')
                    elif key == ord('-'):
                        handle_key('-')
                    elif key == ord('v'):
                        handle_key('v')
                    elif key == ord('V'):
                        handle_key('V')
                    elif key == ord('d'):
                        handle_key('d')
                    elif key == ord('D'):
                        handle_key('D')
                    elif key == ord(' '):
                        handle_key(' ')
                
                # Start timing the frame
                if has_fps_tracker:
                    fps_tracker.start_segment("Simulation")
                
                # Step the simulation
                if has_fps_tracker:
                    fps_tracker.start_segment("Step")
                simulator.step()
                
                # Render the visualization
                if has_fps_tracker:
                    fps_tracker.start_segment("Render")
                
                # Get visualization
                try:
                    if vis_type == 'dye':
                        vis_np = simulator.get_dye_visualization()
                        print("Using dye visualization")
                    elif vis_type == 'density':
                        vis_np = simulator.get_density_visualization()
                        print("Using density visualization")
                    elif vis_type == 'velocity':
                        vis_np = simulator.get_velocity_visualization()
                        print("Using velocity visualization")
                    elif vis_type == 'combined':
                        dye_vis = simulator.get_dye_visualization()
                        vel_vis = simulator.get_velocity_visualization()
                        vis_np = dye_vis * blend_factor + vel_vis * (1 - blend_factor)
                        print(f"Using combined visualization with blend factor {blend_factor}")
                    
                    # Convert tensor to numpy array for rendering
                    vis_np = vis_np.cpu().numpy()
                    
                    # Check for NaN or infinity values
                    if np.isnan(vis_np).any() or np.isinf(vis_np).any():
                        print(f"Warning: NaN or Inf values detected in visualization. Replacing with zeros.")
                        vis_np = np.nan_to_num(vis_np, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Ensure values are in valid range
                    vis_np = np.clip(vis_np, 0, 1)
                    
                    # Scale to 0-255 range for rendering
                    vis_np = (vis_np * 255).astype(np.uint8)
                    
                except Exception as e:
                    print(f"Error getting visualization: {e}")
                    # Create a fallback visualization
                    vis_np = np.ones((height, width, 3), dtype=np.uint8) * 128
                    # Add text to indicate error
                    cv2.putText(vis_np, f"Visualization Error: {vis_type}", (50, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Update pygame control window with current mode
                pygame_window.fill((0, 0, 0))
                
                # Use a larger font for better visibility
                title_font = pygame.font.Font(None, 48)
                font = pygame.font.Font(None, 36)
                small_font = pygame.font.Font(None, 24)
                
                # Draw title
                title = title_font.render("NAVIER-STOKES CONTROL", True, (255, 255, 0))
                pygame_window.blit(title, (10, 10))
                
                # Draw current mode
                mode_text = font.render(f"Mode: {vis_type.upper()}", True, (255, 255, 255))
                pygame_window.blit(mode_text, (10, 70))
                
                # Draw parameters
                if vis_type == 'combined':
                    blend_text = font.render(f"Blend: {blend_factor:.2f}", True, (255, 255, 255))
                    pygame_window.blit(blend_text, (10, 110))
                    y_offset = 150
                else:
                    y_offset = 110
                
                visc_text = font.render(f"Viscosity: {simulator.viscosity:.6f}", True, (255, 255, 255))
                pygame_window.blit(visc_text, (10, y_offset))
                
                diff_text = font.render(f"Diffusion: {simulator.diffusion:.6f}", True, (255, 255, 255))
                pygame_window.blit(diff_text, (10, y_offset + 40))
                
                # Draw key feedback if available
                if key_feedback and time.time() - last_key_time < 2.0:
                    # Calculate alpha based on time (fade out)
                    alpha = min(255, int(255 * (1.0 - (time.time() - last_key_time) / 2.0)))
                    feedback_color = (0, 255, 0, alpha)  # Green with alpha
                    
                    feedback_text = font.render(key_feedback, True, (0, 255, 0))
                    pygame_window.blit(feedback_text, (10, y_offset + 80))
                
                # Draw controls
                controls_y = y_offset + 130
                help_text = small_font.render("CONTROLS:", True, (200, 200, 200))
                pygame_window.blit(help_text, (10, controls_y))
                
                controls = [
                    "MOUSE: Left-click for velocity, Right-click for dye",
                    "1-4: Change visualization mode",
                    "+/-: Adjust blend factor",
                    "V/v: Increase/decrease viscosity",
                    "D/d: Increase/decrease diffusion",
                    "Space: Add velocity burst",
                    "ESC: Quit"
                ]
                
                for i, control in enumerate(controls):
                    control_text = small_font.render(control, True, (200, 200, 200))
                    pygame_window.blit(control_text, (20, controls_y + 25 + i * 20))
                
                # Update the pygame display
                pygame.display.flip()
                
                # Render with lunar_tools if available
                if use_lunar_tools:
                    try:
                        renderer.render(vis_np)
                    except Exception as e:
                        if frame_count == 0:
                            print(f"Error with lunar_tools renderer: {e}")
                            print("Falling back to OpenCV renderer")
                        use_lunar_tools = False
                
                # Always show in OpenCV window as a backup
                try:
                    # Add text overlay to OpenCV window to show current mode
                    vis_np_with_text = vis_np.copy()
                    cv2.putText(vis_np_with_text, f"Mode: {vis_type.upper()}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Show key feedback if available
                    if key_feedback and time.time() - last_key_time < 2.0:
                        cv2.putText(vis_np_with_text, key_feedback, (10, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Navier-Stokes Fluid Simulation", vis_np_with_text)
                except Exception as e:
                    if frame_count == 0:
                        print(f"Error with OpenCV renderer: {e}")
                
                # Display FPS if available
                if has_fps_tracker:
                    fps_tracker.print_fps()
                else:
                    # Simple FPS calculation
                    if frame_count % 60 == 0:
                        current_fps = clock.get_fps()
                        print(f"FPS: {current_fps:.1f}")
                
                # Control frame rate
                clock.tick(60)  # Limit to 60 FPS
                frame_count += 1
            
            # Clean up
            print("Cleaning up resources...")
            try:
                cv2.destroyAllWindows()
                print("OpenCV windows closed")
            except Exception as e:
                print(f"Error closing OpenCV windows: {e}")
                
            try:
                pygame.quit()
                print("Pygame quit successfully")
            except Exception as e:
                print(f"Error quitting pygame: {e}")
            
            print("Simulation ended")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("This example requires lunar_tools and pygame to be installed.")
            print("Please install them or run your own visualization loop using the NavierStokesSimulator.")
        except Exception as e:
            print(f"Error running simulation: {e}")
            import traceback
            traceback.print_exc()
            # Clean up
            try:
                cv2.destroyAllWindows()
            except:
                pass
            try:
                pygame.quit()
            except:
                pass


# If this file is run directly, run the example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Navier-Stokes fluid simulation visualization')
    parser.add_argument('--width', type=int, default=1024, help='Width of the simulation window')
    parser.add_argument('--height', type=int, default=768, help='Height of the simulation window')
    parser.add_argument('--fullscreen', action='store_true', help='Run in fullscreen mode')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the simulation on (cuda:0 or cpu)')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import torch
        import pygame
        import lunar_tools as lt
        
        # Check if CUDA is available, if not use CPU
        if not torch.cuda.is_available() and args.device.startswith('cuda'):
            print("CUDA not available, using CPU instead.")
            args.device = 'cpu'
        
        print("Starting Navier-Stokes fluid simulation visualization...")
        print("Press ESC to exit")
        
        NavierStokesSimulator.run_example(
            width=args.width,
            height=args.height,
            fullscreen=args.fullscreen,
            device=args.device
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("\nThis example requires the following dependencies:")
        print("- PyTorch: pip install torch")
        print("- Pygame: pip install pygame")
        print("- Lunar Tools: Follow installation instructions for your project")

