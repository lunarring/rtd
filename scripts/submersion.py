# TODO:
# client-server
# oscillator mods
#  autoprompt cycler
#  nomotion detector

from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt
from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.optical_flow import OpticalFlowEstimator
from rtd.utils.posteffect import Posteffect
from rtd.utils.audio_detector import AudioDetector
from rtd.utils.oscillators import Oscillator
from rtd.utils.prompt_provider import (
    PromptProviderMicrophone,
    PromptProviderTxtFile,
)
import time
import numpy as np
from rtd.utils.frame_interpolation import AverageFrameInterpolator
import torch


def get_sample_shape_unet(coord, noise_resolution_h, noise_resolution_w):
    channels = 640 if coord[0] == "e" else 1280 if coord[0] == "b" else 640
    if coord[0] == "e":
        coef = float(2 ** int(coord[1]))
        shape = [1, channels, int(np.ceil(noise_resolution_h / coef)), int(np.ceil(noise_resolution_w / coef))]
    elif coord[0] == "b":
        shape = [1, channels, int(np.ceil(noise_resolution_h / 4)), int(np.ceil(noise_resolution_w / 4))]
    else:
        coef = float(2 ** (2 - int(coord[1])))
        shape = [1, channels, int(np.ceil(noise_resolution_h / coef)), int(np.ceil(noise_resolution_w / coef))]
    return shape


if __name__ == "__main__":
    height_diffusion = int((384 + 96) * 1.0)  # 12 * (384 + 96) // 8
    width_diffusion = int((512 + 128) * 1.0)  # 12 * (512 + 128) // 8
    height_render = 1080
    width_render = 1920
    n_frame_interpolations: int = 5
    shape_hw_cam = (576, 1024)
    do_compile = True
    do_diffusion = True
    do_fullscreen = True
    do_enable_dynamic_processor = True

    device = "cuda:0"
    img_diffusion = None

    dynamic_processor = DynamicProcessor()

    if do_diffusion:
        device = "cuda:0"
    else:
        device = "cpu"

    init_prompt = "Human figuring painted with the fast DMT splashes of light, colorful traces of light"
    init_prompt = "Rare colorful flower petals, intricate blue interwoven patterns of exotic flowers"
    # init_prompt = 'Trippy and colorful long neon forest leaves and folliage fractal merging'
    init_prompt = "Dancing people full of glowing neon nerve fibers and filamenets"
    init_prompt = "glowing digital fire full of glitches and neon matrix powerful fire glow and plasma"
    init_prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'

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
        embeds_source = em.clone_embeddings(embeds)
        embeds_target = em.clone_embeddings(embeds)
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

    # Initialize modulations dictionary and noise
    modulations = {}
    modulations_noise = {}
    noise_resolution_h = height_diffusion // 8  # Latent height
    noise_resolution_w = width_diffusion // 8  # Latent width
    for layer in ["e0", "e1", "e2", "e3", "b0", "d0", "d1", "d2", "d3"]:
        shape = get_sample_shape_unet(layer, noise_resolution_h, noise_resolution_w)
        modulations_noise[layer] = torch.randn(shape, device=device).half()

    acid_processor = AcidProcessor(
        height_diffusion=height_diffusion,
        width_diffusion=width_diffusion,
        device=device,
    )
    speech_detector = lt.Speech2Text()
    prompt_provider_microphone = PromptProviderMicrophone()
    prompt_provider_txt_file = PromptProviderTxtFile("materials/prompts/good_prompts_wl_community.txt")
    opt_flow_estimator = OpticalFlowEstimator(use_ema=False)

    posteffect_processor = Posteffect()

    # audio volume level detector
    audio_detector = AudioDetector()

    # initialize effect value oscillator
    oscillator = Oscillator()

    # Initialize FPS tracking
    fps_tracker = lt.FPSTracker()

    do_prompt_change = False
    fract_blend_embeds = 0.0

    while True:
        t_processing_start = time.time()
        # bools
        new_prompt_mic_unmuter = meta_input.get(akai_lpd8="A1", akai_midimix="A3", button_mode="held_down")
        prompt_transition_time = meta_input.get(akai_lpd8="G1", akai_midimix="A1", val_min=1, val_max=20, val_default=8.0)
        do_cycle_prompt_from_file = meta_input.get(akai_lpd8="C0", akai_midimix="A4", button_mode="pressed_once")

        dyn_prompt_mic_unmuter = meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")
        do_dynamic_processor = meta_input.get(akai_lpd8="B0", akai_midimix="B4", button_mode="toggle", val_default=False)
        dyn_prompt_restore_backup = meta_input.get(akai_midimix="F3", button_mode="released_once")
        dyn_prompt_del_current = meta_input.get(akai_midimix="F4", button_mode="released_once")

        do_human_seg = meta_input.get(akai_lpd8="B1", akai_midimix="E3", button_mode="toggle", val_default=True)
        do_acid_wobblers = meta_input.get(akai_lpd8="C1", akai_midimix="D3", button_mode="toggle", val_default=False)
        do_infrared_colorize = meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False)
        do_debug_seethrough = meta_input.get(akai_lpd8="D1", akai_midimix="H3", button_mode="toggle", val_default=False)
        do_postproc = meta_input.get(akai_midimix="G3", button_mode="toggle", val_default=True)
        do_audio_modulation = meta_input.get(akai_midimix="D4", button_mode="toggle", val_default=False)
        do_param_oscillators = meta_input.get(akai_midimix="C3", button_mode="toggle", val_default=False)

        do_optical_flow = meta_input.get(akai_midimix="C4", button_mode="toggle", val_default=True)
        do_postproc = meta_input.get(akai_midimix="E4", button_mode="toggle", val_default=True)

        # floats
        acid_strength = meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.05)
        acid_strength_foreground = meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.05)
        coef_noise = meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=0.3, val_default=0.05)
        zoom_factor = meta_input.get(akai_lpd8="F1", akai_midimix="H2", val_min=0.5, val_max=1.5, val_default=1.0)
        x_shift = int(meta_input.get(akai_midimix="H0", val_min=-50, val_max=50, val_default=0))
        y_shift = int(meta_input.get(akai_midimix="H1", val_min=-50, val_max=50, val_default=0))
        color_matching = meta_input.get(akai_lpd8="G0", akai_midimix="G0", val_min=0, val_max=1, val_default=0.5)
        brightness = meta_input.get(akai_midimix="A0", val_min=0.0, val_max=2, val_default=1.0)
        rotation_angle = meta_input.get(akai_midimix="D0", val_min=-30, val_max=30, val_default=0)

        # Modulation controls
        mod_samp = meta_input.get(akai_midimix="F2", val_min=0, val_max=10, val_default=0)
        mod_emb = meta_input.get(akai_midimix="F1", val_min=0, val_max=10, val_default=2)

        # Set up modulations dictionary
        modulations["modulations_noise"] = modulations_noise
        modulations["b0_samp"] = torch.tensor(mod_samp, device=device)
        modulations["e2_samp"] = torch.tensor(mod_samp, device=device)
        modulations["b0_emb"] = torch.tensor(mod_emb, device=device)
        modulations["e2_emb"] = torch.tensor(mod_emb, device=device)

        # Update DiffusionEngine modulations
        de_img.modulations = modulations

        # dynamic_func_coef1 = meta_input.get(akai_midimix="F0", val_min=0, val_max=1, val_default=0.5)
        # dynamic_func_coef2 = meta_input.get(akai_midimix="F1", val_min=0, val_max=1, val_default=0.5)
        # dynamic_func_coef3 = meta_input.get(akai_midimix="F2", val_min=0, val_max=1, val_default=0.5)

        #  postproc control
        postproc_func_coef1 = meta_input.get(akai_lpd8="H0", akai_midimix="G1", val_min=0, val_max=1, val_default=0.5)
        postproc_func_coef2 = meta_input.get(akai_lpd8="H1", akai_midimix="G2", val_min=0, val_max=1, val_default=0.5)
        postproc_mod_button1 = meta_input.get(akai_midimix="G4", button_mode="toggle", val_default=True)

        #  oscillator-based control
        if do_param_oscillators:
            do_cycle_prompt_from_file = oscillator.get("prompt_cycle", 60, 0, 1, "trigger")
            acid_strength = oscillator.get("acid_strength", 30, 0, 0.5, "continuous")
            coef_noise = oscillator.get("coef_noise", 10, 0, 1.15, "continuous")
            postproc_func_coef1 = oscillator.get("postproc_func_coef1", 120, 0.25, 1, "continuous")
            postproc_func_coef2 = oscillator.get("postproc_func_coef2", 180, 0, 0.5, "continuous")

        #  sound-based control
        if do_audio_modulation:
            sound_volume = audio_detector.get_last_volume()
            #  print(f"Sound volume: {sound_volume}")
        else:
            sound_volume = 0

        do_blur = False
        do_acid_tracers = True

        # if not do_enable_dynamic_processor:
        #     do_dynamic_processor = False

        # if do_compile and do_dynamic_processor:
        #     print(f'dynamic processor is currently not compatible with compile mode')
        #     do_dynamic_processor = False

        new_diffusion_prompt_available_from_mic = prompt_provider_microphone.handle_unmute_button(new_prompt_mic_unmuter)

        if new_diffusion_prompt_available_from_mic:
            current_prompt = prompt_provider_microphone.get_current_prompt()
            do_prompt_change = True
            # print(f"New prompt: {current_prompt}")
            # if do_diffusion:
            #     embeds = em.encode_prompt(current_prompt)
            #     de_img.set_embeddings(embeds)

        if do_cycle_prompt_from_file:
            prompt_provider_txt_file.handle_prompt_cycling_button(do_cycle_prompt_from_file)
            do_prompt_change = True
            current_prompt = prompt_provider_txt_file.get_current_prompt()
            # print(f"New prompt: {current_prompt}")
            # if do_diffusion:
            #     embeds = em.encode_prompt(current_pro
            # mpt)
            #     de_img.set_embeddings(embeds)

        # if we get new prompt: set current embeds as source embeds, get target embeds
        if do_prompt_change and do_diffusion:
            print(f"New prompt: {current_prompt}")
            embeds_source = em.clone_embeddings(embeds)
            embeds_target = em.encode_prompt(current_prompt)
            do_prompt_change = False
            # Reset the blend fraction when starting a new transition
            fract_blend_embeds = 0.0
            # Store the time when transition started
            transition_start_time = time.time()

        # Calculate the blend fraction based on elapsed time and transition duration
        if fract_blend_embeds < 1.0:
            elapsed_time = time.time() - transition_start_time if "transition_start_time" in locals() else 0
            # Calculate fraction based on elapsed time and total transition time
            fract_blend_embeds = min(elapsed_time / prompt_transition_time, 1.0)

            # Blend embeds based on the calculated fraction
            embeds = em.blend_two_embeds(embeds_source, embeds_target, fract_blend_embeds)
            de_img.set_embeddings(embeds)

        #
        img_cam = cam.get_img()

        fps_tracker.start_segment("Optical Flow")
        if do_optical_flow:
            opt_flow = opt_flow_estimator.get_optflow(img_cam.copy(), low_pass_kernel_size=55, window_length=55)
        else:
            opt_flow = None

        fps_tracker.start_segment("Input Image Proc")
        # Start timing image processing
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_resizing_factor_humanseg(0.4)
        input_image_processor.set_blur(do_blur)
        input_image_processor.set_brightness(brightness)
        input_image_processor.set_infrared_colorize(do_infrared_colorize)
        img_proc, human_seg_mask = input_image_processor.process(img_cam.copy())

        if not do_human_seg:
            human_seg_mask = np.ones_like(img_proc).astype(np.float32) / 255

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
        acid_processor.set_rotation_angle(rotation_angle)
        img_acid = acid_processor.process(img_proc, human_seg_mask)

        # Start timing diffusion
        de_img.set_input_image(img_acid)
        de_img.set_guidance_scale(0.5)
        de_img.set_strength(1 / de_img.num_inference_steps + 0.00001)

        fps_tracker.start_segment("Diffusion")
        img_diffusion = np.array(de_img.generate())

        # apply posteffect
        if do_postproc:
            if do_enable_dynamic_processor:
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
                    np.flip(img_diffusion.astype(np.float32), axis=1).copy(),
                    human_seg_mask.astype(np.float32) / 255,
                    opt_flow,
                    postproc_func_coef1,
                )
                update_img = np.clip(img_proc, 0, 255).astype(np.uint8)
                output_to_render = update_img

            else:
                fps_tracker.start_segment("Postprocessor")
                if opt_flow is not None:
                    output_to_render, update_img = posteffect_processor.process(
                        img_diffusion,
                        human_seg_mask.astype(np.float32) / 255,
                        opt_flow,
                        postproc_func_coef1,
                        postproc_func_coef2,
                        postproc_mod_button1,
                        sound_volume,
                    )
                else:
                    output_to_render = img_diffusion
                    update_img = img_diffusion
        else:
            update_img = img_diffusion
            output_to_render = img_diffusion

        acid_processor.update(update_img)

        # fps_tracker.start_segment("Interpolation")
        # interpolated_frames = frame_interpolator.interpolate(img_diffusion)

        fps_tracker.start_segment("Rendering")
        t_processing = time.time() - t_processing_start
        # for frame in interpolated_frames:
        renderer.render(img_proc if do_debug_seethrough else output_to_render)

        # Update and display FPS (this will also handle the last segment timing)
        fps_tracker.print_fps()
