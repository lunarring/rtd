from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
import numpy as np
from rtd.utils.input_image import InputImageProcessor
import sys

if __name__ == "__main__":
    # Example for overriding kwargs
    height_diffusion = 512
    width_diffusion = 512
    shape_hw = (576,1024)

    de_img = DiffusionEngine(use_image2image=True, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
    em = EmbeddingsMixer(de_img.pipe)
    embeds = em.encode_prompt("photo of a new house")
    de_img.set_embeddings(embeds)

    renderer = lt.Renderer(width=width_diffusion, height=height_diffusion, backend='opencv', do_fullscreen=False)
    cam = lt.WebCam(shape_hw=shape_hw)
    input_image_processor = InputImageProcessor()


    while True:
        cam_img = cam.get_img()
        cam_img, human_segmmask = input_image_processor.process(cam_img)
        de_img.set_input_image(cam_img)
        img = de_img.generate()
        renderer.render(img)


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