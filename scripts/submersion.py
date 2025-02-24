from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.optical_flow import OpticalFlowEstimator
from rtd.utils.posteffect import Posteffect
from rtd.utils.prompt_provider import (
    PromptProviderMicrophone,
    PromptProviderTxtFile,
)
import time
import numpy as np
from rtd.utils.frame_interpolation import AverageFrameInterpolator


if __name__ == "__main__":
    height_diffusion = (384 + 96)*1  # 12 * (384 + 96) // 8
    width_diffusion = (512 + 128)*1  # 12 * (512 + 128) // 8
    height_render = 1080
    width_render = 1920
    n_frame_interpolations: int = 5
    shape_hw_cam = (576, 1024)
    do_compile = True
    do_diffusion = True
    do_fullscreen = True

    device = "cuda:0"
    img_diffusion = None

    dynamic_processor = DynamicProcessor()

    if do_diffusion:
        device = "cuda:0"
    else:
        device = "cpu"

    init_prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'
    init_prompt = 'Human figuring painted with the fast DMT splashes of light, colorful traces of light'
    # init_prompt = 'Rare colorful flower petals, intricate blue interwoven patterns of exotic flowers'
    # init_prompt = 'Trippy and colorful long neon forest leaves and folliage fractal merging'
    # init_prompt = 'Dancing people full of glowing neon nerve fibers and filamenets'

    meta_input = lt.MetaInput()
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
    speech_detector = lt.Speech2Text()
    prompt_provider = PromptProviderMicrophone()

    opt_flow_estimator = OpticalFlowEstimator(use_ema=False)

    posteffect_processor = Posteffect()

    # Initialize FPS tracking
    fps_tracker = lt.FPSTracker()

    # Frame interpolator for smooth transitions
    # frame_interpolator = AverageFrameInterpolator(num_frames=n_frame_interpolations)
    while True:
        t_processing_start = time.time()
        # bools
        dyn_prompt_mic_unmuter = meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")
        new_prompt_mic_unmuter = meta_input.get(akai_lpd8="A1", akai_midimix="A3", button_mode="held_down")
        do_dynamic_processor = meta_input.get(akai_lpd8="B0", akai_midimix="B4", button_mode="toggle", val_default=False)
        do_human_seg = meta_input.get(akai_lpd8="B1", akai_midimix="E3", button_mode="toggle", val_default=True)
        cycle_prompt = meta_input.get(akai_lpd8="C0", akai_midimix="C3", button_mode="pressed_once")
        do_acid_wobblers = meta_input.get(akai_lpd8="C1", akai_midimix="D3", button_mode="toggle", val_default=False)
        do_infrared_colorize = meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False)
        do_debug_seethrough = meta_input.get(akai_lpd8="D1", akai_midimix="H3", button_mode="toggle", val_default=False)
        do_postproc = meta_input.get(akai_midimix="G3", button_mode="toggle", val_default=True)

        dyn_prompt_restore_backup = meta_input.get(akai_midimix="F3", button_mode="released_once")
        dyn_prompt_del_current = meta_input.get(akai_midimix="F4", button_mode="released_once")

        # floats
        acid_strength = meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.05)
        acid_strength_foreground = meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.05)
        coef_noise = meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=1.0, val_default=0.05)
        zoom_factor = meta_input.get(akai_lpd8="F1", akai_midimix="A2", val_min=0.5, val_max=1.5, val_default=1.0)
        x_shift = int(meta_input.get(akai_lpd8="H0", akai_midimix="H0", val_min=-50, val_max=50, val_default=0))
        y_shift = int(meta_input.get(akai_lpd8="H1", akai_midimix="H1", val_min=-50, val_max=50, val_default=0))
        color_matching = meta_input.get(akai_lpd8="G0", akai_midimix="G0", val_min=0, val_max=1, val_default=0.5)
        optical_flow_low_pass_kernel_size = int(meta_input.get(akai_midimix="B1", val_min=0, val_max=100, val_default=55))

        dynamic_func_coef1 = meta_input.get(akai_midimix="F0", val_min=0, val_max=1, val_default=0.5)
        dynamic_func_coef2 = meta_input.get(akai_midimix="F1", val_min=0, val_max=1, val_default=0.5)
        dynamic_func_coef3 = meta_input.get(akai_midimix="F2", val_min=0, val_max=1, val_default=0.5)

        postproc_func_coef1 = meta_input.get(akai_midimix="G1", val_min=0, val_max=1, val_default=0.5)
        postproc_func_coef2 = meta_input.get(akai_midimix="G2", val_min=0, val_max=1, val_default=0.5)

        do_blur = False
        do_acid_tracers = True

        if do_dynamic_processor:
            assert not do_compile

        new_diffusion_prompt_available = prompt_provider.handle_unmute_button(new_prompt_mic_unmuter)
        # prompt_provider.handle_prompt_cycling_button(cycle_prompt)

        if new_diffusion_prompt_available:
            current_prompt = prompt_provider.get_current_prompt()
            print(f"New prompt: {current_prompt}")
            if do_diffusion:
                embeds = em.encode_prompt(current_prompt)
                de_img.set_embeddings(embeds)

        img_cam = cam.get_img()

        fps_tracker.start_segment("Optical Flow")
        opt_flow = opt_flow_estimator.get_optflow(img_cam.copy(), 
                                                  low_pass_kernel_size=optical_flow_low_pass_kernel_size, window_length=55)

        fps_tracker.start_segment("Input Image Proc")
        # Start timing image processing
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_resizing_factor_humanseg(0.4)
        input_image_processor.set_blur(do_blur)
        input_image_processor.set_infrared_colorize(do_infrared_colorize)
        img_proc, human_seg_mask = input_image_processor.process(img_cam)

        if not do_human_seg:
            human_seg_mask = np.ones_like(img_proc).astype(np.float32) / 255

        new_dynamic_prompt_available = speech_detector.handle_unmute_button(dyn_prompt_mic_unmuter)

        if new_dynamic_prompt_available:
            dynamic_processor.update_protoblock(speech_detector.transcript)
        if do_dynamic_processor and img_diffusion is not None:
            fps_tracker.start_segment("Dynamic Proc")
            if dyn_prompt_restore_backup:
                dynamic_processor.restore_backup()
            if dyn_prompt_del_current:
                dynamic_processor.delete_current_fn_func()
            img_proc = dynamic_processor.process(
                np.flip(img_proc, axis=1).astype(np.float32),
                human_seg_mask.astype(np.float32) / 255,
                np.flip(img_diffusion.astype(np.float32), axis=1).copy(),
                opt_flow,
                dynamic_func_coef1,
            )
            img_acid = np.clip(img_proc, 0, 255).astype(np.uint8)
        else:
            fps_tracker.start_segment("Acid Proc")
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
        de_img.set_input_image(img_acid)
        de_img.set_guidance_scale(0.5)
        de_img.set_strength(1 / de_img.num_inference_steps + 0.00001)

        fps_tracker.start_segment("Diffusion")
        img_diffusion = np.array(de_img.generate())

        # apply posteffect
        if do_postproc:
            fps_tracker.start_segment("Postprocessor")
            if opt_flow is not None:
                img_diffusion = posteffect_processor.process(img_diffusion, 
                                                        human_seg_mask.astype(np.float32) / 255, opt_flow,
                                                        postproc_func_coef1, postproc_func_coef2)

        acid_processor.update(img_diffusion)

        fps_tracker.start_segment("Interpolation")
        # interpolated_frames = frame_interpolator.interpolate(img_diffusion)

        fps_tracker.start_segment("Rendering")
        t_processing = time.time() - t_processing_start
        # for frame in interpolated_frames:
        renderer.render(img_proc if do_debug_seethrough else img_diffusion)

        # Update and display FPS (this will also handle the last segment timing)
        fps_tracker.print_fps()
