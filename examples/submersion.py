from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt

# Example for overriding kwargs
height_diffusion = 512
width_diffusion = 512
de_img = DiffusionEngine(use_image2image=True, height_diffusion_desired=height_diffusion, width_diffusion_desired=width_diffusion)
em = EmbeddingsMixer(de_img.pipe)
embeds = em.encode_prompt("photo of a new house")
de_img.set_embeddings(embeds)

renderer = lt.Renderer(width=width_diffusion, height=height_diffusion, backend='opencv', do_fullscreen=False)
cam = lt.WebCam()
# midi_input = lt.MidiInput(device_name="akai_midimix")

import sys
if len(sys.argv) > 1 and sys.argv[1] == "processor":
    import numpy as np
    from rtd.utils.input_image import InputImageProcessor
    # Instantiate the processor and force a valid blur setting to avoid runtime errors
    processor = InputImageProcessor()
    processor.set_blur_size(3)
    # Create a dummy image of shape 256x256x3
    dummy_img = np.random.randint(0,256,(256,256,3), dtype=np.uint8)
    processed_img, seg_mask = processor.process(dummy_img)
    print("Integration Demo:")
    print("Processed image shape:", processed_img.shape)
    if seg_mask is not None:
         print("Segmentation mask shape:", seg_mask.shape)
    else:
         print("No segmentation mask returned.")
    sys.exit(0)

# acid_process = InputImageProcessor()

while True:
    cam_img = cam.get_img()
    de_img.set_input_image(cam_img)
    img = de_img.generate()
    renderer.render(img)
