from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
import numpy as np
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.fps_tracker import FPSTracker
from rtd.utils.prompt_provider import PromptProviderMicrophone
import sys

if __name__ == "__main__":
    height_diffusion = 384
    width_diffusion = 512
    height_render = 1080
    width_render = 1920
    shape_hw_cam = (576,1024)
    do_compile = False

    akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")
    de_img = DiffusionEngine(use_image2image=True, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion, do_compile=do_compile)
    em = EmbeddingsMixer(de_img.pipe)
    embeds = em.encode_prompt("photo of a cat")
    de_img.set_embeddings(embeds)

    renderer = lt.Renderer(width=width_render, height=height_render, backend='opencv', do_fullscreen=False)
    cam = lt.WebCam(shape_hw=shape_hw_cam)
    input_image_processor = InputImageProcessor()
    input_image_processor.set_flip(do_flip=True, flip_axis=1)

    acid_processor = AcidProcessor(height_diffusion=height_diffusion, width_diffusion=width_diffusion)

    prompt_provider = PromptProviderMicrophone(init_prompt="Image of a cat")

    # Initialize FPS tracking
    fps_tracker = FPSTracker()

    while True:
        do_human_seg = akai_lpd8.get("A0", button_mode='toggle', val_default=True) 
        mic_button_state = akai_lpd8.get("A1", button_mode='held_down') 
        do_blur = akai_lpd8.get("B0", button_mode='toggle', val_default=True) 
        do_acid_tracers = akai_lpd8.get("B1", button_mode='toggle', val_default=True) 
        do_debug_seethrough = akai_lpd8.get("D1", button_mode='toggle', val_default=False)
        acid_strength = akai_lpd8.get("E0", val_min=0, val_max=1.0, val_default=0.1)
        acid_strength_foreground = akai_lpd8.get("E1", val_min=0, val_max=1.0, val_default=0.1)
        coef_noise = akai_lpd8.get("F0", val_min=0, val_max=1.0, val_default=0.15) 
        zoom_factor = akai_lpd8.get("F1", val_min=0.5, val_max=1.5, val_default=1.0) 

        prompt_provider.handle_mic_button(mic_button_state)

        if prompt_provider.new_prompt_available():
            current_prompt = prompt_provider.get_current_prompt()
            print(f"New prompt: {current_prompt}")
            embeds = em.encode_prompt(current_prompt)
            de_img.set_embeddings(embeds)
            
        img_cam = cam.get_img()
        # Start timing image processing
        fps_tracker.start_segment("Image Proc")
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_blur(do_blur)
        img_proc, human_segmmask = input_image_processor.process(img_cam)
        
        # Acid
        acid_processor.set_acid_strength(acid_strength)
        acid_processor.set_coef_noise(coef_noise)
        acid_processor.set_acid_tracers(do_acid_tracers)
        acid_processor.set_acid_strength_foreground(acid_strength_foreground)
        acid_processor.set_zoom_factor(zoom_factor)
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