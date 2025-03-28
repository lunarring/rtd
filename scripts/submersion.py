# TODO:
# possibly short term

# - @alex, why this line? human_seg_mask = final_mask * 255
# - AR knob
# - @alex review and improve optical flow mask
# - optical flow acid
# - reconfigure akai midimix (use sliders)
# - investigate better noise
#   - noise measurement (experiments)
#   - noise reduction:
#     - median (blur)
#     - kalman filter
#     - other filters (using optflow?)?
#     - denoising
# - automatic prompt injection
# - understand mem acid better
# - smooth continuation mode
# - objects floating around or being interactive
# - display prompts option

# long term
# - parallelization and stitching
# - image to guiding prompt (engagement with image)
# -

# done (review)
# - smooth prompt blending A -> B

from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
import lunar_tools as lt

# from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.optical_flow import OpticalFlowEstimator
from rtd.utils.posteffect import Posteffect
from rtd.utils.audio_detector import AudioDetector
from rtd.utils.oscillators import Oscillator
from rtd.utils.prompt_provider import (
    PromptProviderMicrophone,
    PromptProviderTxtFile,
    PromptProviderSpeechToText,
)
from rtd.utils.misc_utils import get_repo_path
import time
import numpy as np
from rtd.utils.frame_interpolation import AverageFrameInterpolator
import torch
import os
import sys
import cv2
import socket

from rtd.utils.compression_helpers import send_compressed

vis_llm_prompt = """Your task is to directly translate from poetry into visual descriptions. I give you couple examples and you can do it then for the next one. Here are rules: Your task is to produce a valid and new output. Keep the style and length of the examples in your output. Don't make it longer. Stay within the theme of the examples!

Input: The surface is not motionless. Output: Waves on the surface of endless blue ocean and flickering, silver light 
Input: movement originates from the tides Output: regular movement of waves depends on the (invisible) underwater tides 
Input: pulsations in the bloodstream Output: pulsating liquid flowing through a riverbed
Input: it ripples, reflected on the surface Output: ripples on the surface of magma
Input: First, there's an impulse, a vibration Output: impulse of kinetic energy turning into vibration of particles 
Input: reflection of the waves on sand Output: pattern on the sand resembles the shape of a wave
Input: dunes in slow-motion Output: grains of sand carried by the wind are forming dunes
Input: birds respond to magnetic signals Output: flock of birds, lines of flight
Input: reverberating , undulating shapes Output: undulating shapes in an empty space
Input: marked on the sand, on the water, in the air Output: transforming fractal  patterns, green
Input: inscribed into the bodies of the rocks Output: fossils and geological lines marked on the rocks
Input: The Metamorphosis of Time Output: transformations of a celestial mass happening across deep time
Input: on the still mirror of the lake Output: still lake hidden in a forest, glossy surface
Input: whispers of a breeze, water spirits murmuring their songs Output: breeze moving a surface of water
Input: break the stillness of the air Output: subtle frequencies disturbing the  air
Input: based on frequencies and undulations of the sound Output: sound undulating in a void, bright light
Input: the lullaby Output: silent landscape, starry sky
Input: the poem from the past appearing suddenly in your memory Output: blurry image of a glacier
Input: a turbulence Output: image of a gray sky, turbulence
Input: Wind is hitting my skin and I imagine its waves spreading into a kaleidoscope of light Output: Wind blows spreading into a kaleidoscope of light
Input: particles rolling on the surface Output:  l glass beads rolling on the surface of a petri dish
Input: atoms crashing, constant annihilations and re-productions Output:  elements colliding and dispersing into atoms 
Input: The Pattern of Movement, a ripple Output: trembling  pattern of movement in the kelp forest

The next message I send you will be an Input, and you directly continue after 'Output:'
"""


class TouchDesignerSender:
    def __init__(self, host="localhost", port=9998):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.attempt_reconnect = True
        self.last_reconnect_time = 0
        self.reconnect_interval = 5  # seconds between reconnection attempts

    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.socket.settimeout(0.5)  # Short timeout for connection attempts
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.socket.settimeout(None)  # Reset timeout for normal operation
            print(f"Connected to TouchDesigner at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to TouchDesigner: {e}")
            self.connected = False
            return False

    def send_image(self, image):
        if not self.connected:
            current_time = time.time()
            if self.attempt_reconnect and (current_time - self.last_reconnect_time > self.reconnect_interval):
                self.last_reconnect_time = current_time
                self.connect()

            if not self.connected:
                return False

        try:
            send_compressed(self.socket, image, quality=90)
            return True
        except Exception as e:
            print(f"Error sending to TouchDesigner: {e}")
            self.connected = False
            return False


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
    height_diffusion = int((384 + 96) * 1.5)  # 12 * (384 + 96) // 8
    width_diffusion = int((512 + 128) * 1.5)  # 12 * (512 + 128) // 8
    height_render = 1080
    width_render = 1920
    n_frame_interpolations: int = 5
    shape_hw_cam = (576, 1024)

    touchdesigner_host = "192.168.100.101"  # Change to your TouchDesigner machine's IP
    touchdesigner_port = 9998

    do_realtime_transcription = False
    do_compile = True
    do_diffusion = True
    do_fullscreen = True
    do_enable_dynamic_processor = False
    do_send_to_touchdesigner = False
    do_load_cam_input_from_file = True
    do_save_diffusion_output_to_file = False
    video_file_path_input = get_repo_path("materials/videos/long_cut4.mp4")
    print(video_file_path_input)
    video_file_path_output = "materials/videos/long_cut_diffusion2.mp4"

    device = "cuda:0"
    img_diffusion = None

    if do_enable_dynamic_processor:
        dynamic_processor = DynamicProcessor()

    if do_diffusion:
        device = "cuda:0"
    else:
        device = "cpu"

    init_prompt = "Human figuring painted with the fast DMT splashes of light, colorful traces of light"
    init_prompt = "Rare colorful flower petals, intricate blue interwoven patterns of exotic flowers"
    # init_prompt = 'Trippy and colorful long neon forest leaves and folliage fractal merging'
    init_prompt = "Dancing people full of glowing neon nerve fibers and filamenets"
    # init_prompt = "glowing digital fire full of glitches and neon matrix powerful fire glow and plasma"
    # init_prompt = "glowing digital fire full of glitches and neon matrix powerful fire glow and plasma"
    init_prompt = "The Metamorphosis of Time: Deep below, volcano's asleep, pulsating rhythmically on the still mirror of the lake"

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
    cam = lt.WebCam(shape_hw=shape_hw_cam, do_digital_exposure_accumulation=True, exposure_buf_size=3, cam_id=0)
    cam.do_mirror = False

    # Initialize movie reader if loading from file
    if do_load_cam_input_from_file:
        movie_reader = lt.MovieReader(video_file_path_input)

    # Initialize movie saver if saving output to file
    if do_save_diffusion_output_to_file:
        movie_saver = lt.MovieSaver(video_file_path_output, fps=12)

    input_image_processor = InputImageProcessor(device=device)
    input_image_processor.set_flip(do_flip=True, flip_axis=1)

    if do_send_to_touchdesigner:
        td_sender = TouchDesignerSender(host=touchdesigner_host, port=touchdesigner_port)
        td_sender.connect()

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

    if do_realtime_transcription:
        prompt_provider_stt = PromptProviderSpeechToText(
            init_prompt="A beautiful water sea",
            llm_system_prompt=(vis_llm_prompt),
            temperature=0.3,
        )
    else:
        prompt_provider_mic = PromptProviderMicrophone(init_prompt="A beautiful landscape")

    prompt_provider_txt_file = PromptProviderTxtFile(
        get_repo_path("materials/prompts/gosia_poetry.txt", __file__), mode="sequential"  # Can be "random" or "sequential"
    )
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

    frame_counter = -1
    while True:
        frame_counter += 1
        t_processing_start = time.time()
        # bools
        new_prompt_mic_unmuter = meta_input.get(akai_lpd8="F1", akai_midimix="F3", button_mode="held_down")

        hue_rotation_angle = int(meta_input.get(akai_midimix="A0", val_min=0, val_max=360, val_default=0))
        prompt_transition_time = meta_input.get(akai_lpd8="G1", akai_midimix="F0", val_min=1, val_max=50, val_default=8.0)
        do_cycle_prompt_from_file = meta_input.get(akai_lpd8="C0", akai_midimix="F4", button_mode="pressed_once")

        dyn_prompt_mic_unmuter = False  # meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")
        do_dynamic_processor = False  # meta_input.get(akai_lpd8="B0", akai_midimix="B4", button_mode="toggle", val_default=False)
        dyn_prompt_restore_backup = False  # meta_input.get(akai_midimix="F3", button_mode="released_once")
        dyn_prompt_del_current = False  # meta_input.get(akai_midimix="F4", button_mode="released_once")

        do_human_seg = meta_input.get(akai_lpd8="B1", akai_midimix="A3", button_mode="toggle", val_default=False)
        do_motion_tracking_masking = meta_input.get(akai_midimix="B3", button_mode="toggle", val_default=False)
        do_acid_wobblers = False  # meta_input.get(akai_lpd8="C1", akai_midimix="D3", button_mode="toggle", val_default=False)
        do_infrared_colorize = False  # meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False)
        do_debug_seethrough = meta_input.get(akai_lpd8="D1", akai_midimix="H3", button_mode="toggle", val_default=False)
        do_audio_modulation = False  # meta_input.get(akai_midimix="D4", button_mode="toggle", val_default=False)
        do_param_oscillators = False  # meta_input.get(akai_midimix="C3", button_mode="toggle", val_default=False)
        #do_opt_flow_seg = meta_input.get(akai_midimix="G3", button_mode="toggle", val_default=False)
        do_opt_flow_seg = False
        # do_optical_flow = meta_input.get(akai_midimix="C4", button_mode="toggle", val_default=True)
        do_postproc = meta_input.get(akai_midimix="A4", button_mode="toggle", val_default=False)

        do_optical_flow = do_postproc or do_opt_flow_seg
        # floats
        # nmb_inference_steps = meta_input.get(akai_midimix="B0", val_min=2, val_max=10.0, val_default=2.0)
        nmb_inference_steps = 4
        acid_strength = meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.4)
        acid_strength_foreground = meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.4)
        # opt_flow_threshold = meta_input.get(akai_lpd8="E2", akai_midimix="E2", val_min=0, val_max=2, val_default=1)
        opt_flow_threshold = 1
        coef_noise = meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=0.3, val_default=0.08)
        # zoom_factor = meta_input.get(akai_lpd8="F1", akai_midimix="H2", val_min=0.5, val_max=1.5, val_default=1.0)
        zoom_out_factor = meta_input.get(akai_lpd8="F1", akai_midimix="G5", val_min=0, val_max=0.05, val_default=0)
        zoom_in_factor = meta_input.get(akai_lpd8="F1", akai_midimix="H5", val_min=0, val_max=0.05, val_default=0)
        acid_hue_rot = meta_input.get(akai_midimix="B0", val_min=0, val_max=30, val_default=0)
        acid_saturation = meta_input.get(akai_midimix="B1", val_min=-15, val_max=15, val_default=0)
        acid_lightness = meta_input.get(akai_midimix="B2", val_min=-15, val_max=15, val_default=0)
        saturation = meta_input.get(akai_midimix="A1", val_min=0.0, val_max=2.0, val_default=1.0)  # Add saturation control

        if zoom_in_factor * zoom_out_factor == 0:
            zoom_factor = 1 - zoom_in_factor + zoom_out_factor
        else:
            zoom_factor = 1 + oscillator.get("zoom_factor", 1.0 / (0.2 * zoom_in_factor), zoom_out_factor, -zoom_out_factor, "continuous")

        # X shift with oscillation when both directions are active
        x_shift_left = meta_input.get(akai_midimix="G1", val_min=0, val_max=50, val_default=0)
        x_shift_right = meta_input.get(akai_midimix="H1", val_min=0, val_max=50, val_default=0)
        if x_shift_left * x_shift_right == 0:
            x_shift = int(x_shift_right - x_shift_left)
        else:
            x_shift = int(oscillator.get("x_shift", 1.0 / (0.5 * x_shift_right), x_shift_left, -x_shift_right, "continuous"))

        # Y shift with oscillation when both directions are active
        y_shift_up = meta_input.get(akai_midimix="G0", val_min=0, val_max=50, val_default=0)
        y_shift_down = meta_input.get(akai_midimix="H0", val_min=0, val_max=50, val_default=0)
        if y_shift_up * y_shift_down == 0:
            y_shift = int(y_shift_down - y_shift_up)
        else:
            y_shift = int(oscillator.get("y_shift", 1.0 / (0.5 * y_shift_down), y_shift_up, -y_shift_down, "continuous"))

        # Rotation with oscillation when both directions are active
        rotation_left = meta_input.get(akai_midimix="G2", val_min=0, val_max=30, val_default=0)
        rotation_right = meta_input.get(akai_midimix="H2", val_min=0, val_max=30, val_default=0)
        if rotation_left * rotation_right == 0:
            rotation_angle = rotation_right - rotation_left
        else:
            rotation_angle = oscillator.get("rotation_angle", 1.0 / (0.5 * rotation_right), rotation_left, -rotation_right, "continuous")

        color_matching = meta_input.get(akai_lpd8="G0", akai_midimix="A5", val_min=0, val_max=1, val_default=0.5)
        brightness = meta_input.get(akai_midimix="A2", val_min=0.0, val_max=2, val_default=1.0)
        # Add latent acid strength parameter
        # latent_acid_strength = meta_input.get(akai_midimix="D1", val_min=0, val_max=1.0, val_default=0.0)
        latent_acid_strength = 0.0

        # Modulation controls
        # mod_samp = meta_input.get(akai_midimix="F2", val_min=0, val_max=10, val_default=0)
        mod_samp = 0
        mod_emb = meta_input.get(akai_midimix="B5", val_min=0, val_max=10, val_default=5)

        # Set up modulations dictionary
        modulations["modulations_noise"] = modulations_noise
        modulations["b0_samp"] = torch.tensor(mod_samp, device=device)
        modulations["e2_samp"] = torch.tensor(mod_samp, device=device)
        modulations["b0_emb"] = torch.tensor(mod_emb, device=device)
        modulations["e2_emb"] = torch.tensor(mod_emb, device=device)

        # Update DiffusionEngine parameters
        de_img.set_strength(acid_strength)
        # Set latent acid strength
        de_img.set_latent_acid_strength(latent_acid_strength)

        # Update DiffusionEngine modulations
        de_img.modulations = modulations

        # dynamic_func_coef1 = meta_input.get(akai_midimix="F0", val_min=0, val_max=1, val_default=0.5)
        # dynamic_func_coef2 = meta_input.get(akai_midimix="F1", val_min=0, val_max=1, val_default=0.5)
        # dynamic_func_coef3 = meta_input.get(akai_midimix="F2", val_min=0, val_max=1, val_default=0.5)

        #  postproc control
        # postproc_mod_button1 = meta_input.get(akai_midimix="G4", button_mode="toggle", val_default=True)
        postproc_mod_button1 = True
        flow_gain = meta_input.get(akai_lpd8="D0", akai_midimix="D0", val_min=0, val_max=1, val_default=0.3)
        reverb_gain = meta_input.get(akai_lpd8="D1", akai_midimix="D1", val_min=0, val_max=1, val_default=0.3)
        background_image_gain = meta_input.get(akai_midimix="D2", val_min=0, val_max=1, val_default=0.6)
        # postproc_func_coef1 = 0.5
        # postproc_func_coef2 = 0.5
        # postproc_mod_button1 = True
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

        if not do_realtime_transcription:
            new_diffusion_prompt_available_from_mic = prompt_provider_mic.handle_unmute_button(new_prompt_mic_unmuter)

            if new_diffusion_prompt_available_from_mic:
                current_prompt = prompt_provider_mic.get_current_prompt()
                do_prompt_change = True
        else:
            if prompt_provider_stt.new_prompt_available():
                current_prompt = prompt_provider_stt.last_prompt
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
        # Get camera framerate from the camera thread
        camera_fps = cam.get_fps()

        if frame_counter < 10 or not do_load_cam_input_from_file:
            img_cam = cam.get_img()
        else:
            # Get frame from video file instead of webcam
            img_cam = movie_reader.get_next_frame()

            # If we reached the end of the video (get_next_frame returns empty/black frame), reset the video reader
            if img_cam is None or (isinstance(img_cam, np.ndarray) and (img_cam.size == 0 or np.max(img_cam) == 0)):
                print("End of video reached, looping back to the beginning")
                # Re-initialize the movie reader to restart the video
                movie_reader = lt.MovieReader(video_file_path_input)
                # Get the first frame
                img_cam = movie_reader.get_next_frame()

            # Convert BGR to RGB for processing
            if img_cam is not None and img_cam.size > 0:
                img_cam = img_cam[:, :, ::-1].copy()

        img_cam_last = img_cam.copy()

        fps_tracker.start_segment("OptFlow")
        if do_optical_flow:
            opt_flow = opt_flow_estimator.get_optflow(img_cam.copy(), low_pass_kernel_size=55, window_length=55)
        else:
            opt_flow = None

        fps_tracker.start_segment("InImg")
        # Start timing image processing
        input_image_processor.set_human_seg(do_human_seg)
        input_image_processor.set_motion_tracking_masking(do_motion_tracking_masking)
        input_image_processor.set_opt_flow_seg(do_opt_flow_seg)
        input_image_processor.set_resizing_factor_humanseg(0.4)
        input_image_processor.set_blur(do_blur)
        input_image_processor.set_brightness(brightness)
        input_image_processor.set_hue_rotation(hue_rotation_angle)
        input_image_processor.set_infrared_colorize(do_infrared_colorize)
        input_image_processor.set_opt_flow_threshold(opt_flow_threshold)
        input_image_processor.set_saturation(saturation)
        img_proc, human_seg_mask = input_image_processor.process(img_cam, opt_flow)

        # if not do_human_seg and not do_opt_flow_seg: VERY BAD, BREAKS COLOR SCALING!!!
        #     human_seg_mask = np.ones_like(img_proc).astype(np.float32)  # / 255

        fps_tracker.start_segment("Acid")
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
        acid_processor.set_acid_hue_rotation_angle(acid_hue_rot)
        acid_processor.set_acid_saturation_adjustment(acid_saturation)
        acid_processor.set_acid_lightness_adjustment(acid_lightness)
        img_acid = acid_processor.process(img_proc, human_seg_mask)

        # Start timing diffusion
        de_img.set_input_image(img_acid)
        de_img.set_guidance_scale(0.5)
        de_img.set_num_inference_steps(int(nmb_inference_steps))
        de_img.set_strength(1 / de_img.num_inference_steps + 0.00001)

        fps_tracker.start_segment("Diffu")
        img_diffusion = np.array(de_img.generate())

        # apply posteffect
        if do_postproc:
            if human_seg_mask is None:
                hsm = np.ones_like(img_proc).astype(np.float32) / 255
            else:
                hsm = human_seg_mask.astype(np.float32) / 255
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
                    hsm,
                    opt_flow,
                    postproc_func_coef1,
                )
                update_img = np.clip(img_proc, 0, 255).astype(np.uint8)
                output_to_render = update_img

            else:
                fps_tracker.start_segment("Postproc")
                if opt_flow is not None:
                    output_to_render, update_img = posteffect_processor.process(
                        img_diffusion,
                        hsm,
                        opt_flow,
                        flow_gain,
                        reverb_gain,
                        background_image_gain,
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

        fps_tracker.start_segment("Rend")
        t_processing = time.time() - t_processing_start
        # for frame in interpolated_frames:
        renderer.render(img_proc if do_debug_seethrough else output_to_render)

        if do_send_to_touchdesigner:
            td_sender.send_image(output_to_render)

        if do_save_diffusion_output_to_file:
            movie_saver.write_frame(output_to_render)
            if frame_counter >= 512 * 8:
                movie_saver.finalize()
                print(f"Movie saved to {video_file_path_output} after {frame_counter+1} frames")
                do_save_diffusion_output_to_file = False

        # Update and display FPS (this will also handle the last segment timing)
        fps_tracker.print_fps()
