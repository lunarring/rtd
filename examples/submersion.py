from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
from rtd.dynamic_processor.dynamic_processor import DynamicProcessor
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.prompt_provider import (
    PromptProviderMicrophone,
    PromptProviderMicrophoneTxt,
)
import time
from rtd.utils.frame_interpolation import AverageFrameInterpolator


if __name__ == "__main__":
    height_diffusion = 384 + 96  # 12 * (384 + 96) // 8
    width_diffusion = 512 + 128  # 12 * (512 + 128) // 8
    height_render = 1080
    width_render = 1920
    n_frame_interpolations: int = 5
    shape_hw_cam = (576, 1024)
    do_compile = False
    do_diffusion = True
    do_fullscreen = False
    do_dynamic_processor = True
    device = "cuda:0"
    img_diffusion = None

    if do_dynamic_processor:
        assert not do_compile
        dynamic_processor = DynamicProcessor()
    if do_diffusion:
        device = "cuda:0"
    else:
        device = "cpu"

    init_prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'
    # init_prompt = "Normal naked people"

    akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")
    de_img = DiffusionEngine(
        use_image2image=True,
        height_diffusion_desired=height_diffusion,
        width_diffusion_desired=width_diffusion,
        do_compile=do_compile,
        do_diffusion=do_diffusion,
        device=device,
    )
    em = EmbeddingsMixer(de_img.pipe)
    if do_diffusion:
        embeds = em.encode_prompt(init_prompt)
        de_img.set_embeddings(embeds)
    renderer = lt.Renderer(
        width=width_render,
        height=height_render,
        backend="opencv",
        do_fullscreen=do_fullscreen,
    )
    cam = lt.WebCam(shape_hw=shape_hw_cam)
    input_image_processor = InputImageProcessor(device=device)
    input_image_processor.set_flip(do_flip=True, flip_axis=1)

    acid_processor = AcidProcessor(
        height_diffusion=height_diffusion,
        width_diffusion=width_diffusion,
        device=device,
    )

    prompt_provider = PromptProviderMicrophoneTxt(file_path="/home/lugo/git/rtd/prompts/ShamelessVisualization.txt")

    # Initialize FPS tracking
    fps_tracker = lt.FPSTracker()

    # Frame interpolator for smooth transitions
    # frame_interpolator = AverageFrameInterpolator(num_frames=n_frame_interpolations)

    while True:
        t_processing_start = time.time()

        do_human_seg = akai_lpd8.get("A0", button_mode="toggle", val_default=True)
        mic_button_state = akai_lpd8.get("A1", button_mode="held_down")
        cycle_prompt = akai_lpd8.get("C0", button_mode="pressed_once")
        do_blur = True
        inject_dyn_prompt = akai_lpd8.get("B0", button_mode="pressed_once")
        do_acid_tracers = akai_lpd8.get("B1", button_mode="toggle", val_default=True)
        do_acid_wobblers = akai_lpd8.get("C1", button_mode="toggle", val_default=False)
        do_debug_seethrough = akai_lpd8.get("D1", button_mode="toggle", val_default=False)
        acid_strength = akai_lpd8.get("E0", val_min=0, val_max=1.0, val_default=0.11)
        acid_strength_foreground = akai_lpd8.get("E1", val_min=0, val_max=1.0, val_default=0.11)
        coef_noise = akai_lpd8.get("F0", val_min=0, val_max=1.0, val_default=0.15)
        zoom_factor = akai_lpd8.get("F1", val_min=0.5, val_max=1.5, val_default=1.0)
        x_shift = int(akai_lpd8.get("H0", val_min=-50, val_max=50, val_default=0))
        y_shift = int(akai_lpd8.get("H1", val_min=-50, val_max=50, val_default=0))
        color_matching = akai_lpd8.get("G0", val_min=0, val_max=1, val_default=0.5)

        dynamic_func_coef = akai_lpd8.get("G1", val_min=0, val_max=1, val_default=0.5)

        prompt_provider.handle_mic_button(mic_button_state)
        prompt_provider.handle_prompt_cycling_button(cycle_prompt)

        if prompt_provider.new_prompt_available():
            current_prompt = prompt_provider.get_current_prompt()
            print(f"New prompt: {current_prompt}")
            if do_diffusion:
                embeds = em.encode_prompt(current_prompt)
                de_img.set_embeddings(embeds)

        img_cam = cam.get_img()
        # Start timing image processing
        fps_tracker.start_segment("Image Proc")
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_blur(do_blur)
        img_proc, human_seg_mask = input_image_processor.process(img_cam)

        if inject_dyn_prompt:
            dynamic_processor.update_protoblock_voice()
        if do_dynamic_processor and img_diffusion is not None:
            img_acid = dynamic_processor.process(img_cam, human_seg_mask, img_diffusion, dynamic_func_coef=dynamic_func_coef)
            img_proc = img_acid
        else:
            # Acid
            acid_processor.set_acid_strength(acid_strength)
            acid_processor.set_coef_noise(coef_noise)
            acid_processor.set_acid_tracers(do_acid_tracers)
            acid_processor.set_acid_strength_foreground(acid_strength_foreground)
            acid_processor.set_zoom_factor(zoom_factor)
            acid_processor.set_x_shift(x_shift)
            acid_processor.set_y_shift(y_shift)
            acid_processor.set_do_acid_wobblers(do_acid_wobblers)
            acid_processor.set_color_matching(color_matching)
            img_acid = acid_processor.process(img_proc, human_seg_mask)

        # Start timing diffusion
        fps_tracker.start_segment("Acid Proc")
        de_img.set_input_image(img_acid)
        de_img.set_guidance_scale(0.5)
        de_img.set_strength(1 / de_img.num_inference_steps + 0.00001)

        fps_tracker.start_segment("Diffusion")
        img_diffusion = de_img.generate()

        acid_processor.update(img_diffusion)

        fps_tracker.start_segment("Interpolation")
        # interpolated_frames = frame_interpolator.interpolate(img_diffusion)

        fps_tracker.start_segment("Rendering")
        t_processing = time.time() - t_processing_start
        # for frame in interpolated_frames:
        renderer.render(img_proc if do_debug_seethrough else img_diffusion)

        # Update and display FPS (this will also handle the last segment timing)
        # fps_tracker.print_fps()
