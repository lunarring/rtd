from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
import numpy as np
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.fps_tracker import FPSTracker
import sys

if __name__ == "__main__":
    # Example for overriding kwargs
    height_diffusion = 512
    width_diffusion = 512
    shape_hw = (576,1024)


    akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")

    de_img = DiffusionEngine(use_image2image=True, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
    em = EmbeddingsMixer(de_img.pipe)
    embeds = em.encode_prompt("photo of a new house")
    de_img.set_embeddings(embeds)

    renderer = lt.Renderer(width=width_diffusion, height=height_diffusion, backend='opencv', do_fullscreen=False)
    cam = lt.WebCam(shape_hw=shape_hw)
    input_image_processor = InputImageProcessor()
    input_image_processor.set_flip(do_flip=True, flip_axis=1)

    acid_processor = AcidProcessor(height_diffusion=height_diffusion, width_diffusion=width_diffusion)

    # Initialize FPS tracking
    fps_tracker = FPSTracker()

    while True:
        do_human_seg = akai_lpd8.get("A0", button_mode='toggle', val_default=True) # toggle switches the state with every press between on and off
        do_blur = akai_lpd8.get("B0", button_mode='toggle', val_default=True) # toggle switches the state with every press between on and off
        do_debug_seethrough = akai_lpd8.get("D1", button_mode='toggle', val_default=False)
        acid_strength = akai_lpd8.get("E0", val_min=0, val_max=1.0) 
        coef_noise = akai_lpd8.get("E1", val_min=0, val_max=1.0, val_default=0.15) 
        img_cam = cam.get_img()

        # Start timing image processing
        fps_tracker.start_segment("Image Proc")
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_blur(do_blur)
        img_proc, human_segmmask = input_image_processor.process(img_cam)
        
        # Acid
        acid_processor.set_acid_strength(acid_strength)
        acid_processor.set_coef_noise(coef_noise)
        img_acid = acid_processor.process(img_proc)

        # Start timing diffusion
        fps_tracker.start_segment("Acid Proc")
        de_img.set_input_image(img_acid)

        fps_tracker.start_segment("Diffusion")
        img_diffusion  = de_img.generate()

        acid_processor.update(img_diffusion)

        if do_debug_seethrough:
            renderer.render(img_proc)
        else:
            renderer.render(img_diffusion)  

        # Update and display FPS (this will also handle the last segment timing)
        if fps_tracker.update():
            fps_tracker.print_fps()

        # if brightness is not None:
        #     self.iip.set_brightness(brightness)
        # if saturization is not None:
        #     self.iip.set_saturization(saturization)
        # if hue_rotation_angle is not None:
        #     self.iip.set_hue_rotation(hue_rotation_angle)
        # if blur_kernel_size is not None:
        #     self.iip.set_blur_size(blur_kernel_size)
        # if do_blur is not None:
        #     self.iip.set_blur(do_blur)
        # if is_infrared is not None:
        #     self.iip.set_infrared(is_infrared)
        # if do_human_seg is not None:
        #     self.iip.set_human_seg(do_human_seg)
        # if flip_axis:
        #     flip_axis = np.clip(flip_axis, -1, 2)
        #     flip_axis = int(flip_axis)
        #     if flip_axis == -1:
        #         self.iip.set_flip(False, 0)
        #     else:
        #         self.iip.set_flip(True, flip_axis)
        # else:
        #     self.iip.set_flip(False)
        # if resizing_factor_humanseg is not None:
        #     self.iip.set_resizing_factor_humanseg(resizing_factor_humanseg)
        
        
        
        # image = [image, human_segmmask]