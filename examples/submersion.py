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

# acid_process = InputImageProcessor()

while True:
    cam_img = cam.get_img()
    de_img.set_input_image(cam_img)
    img = de_img.generate()
    renderer.render(img)