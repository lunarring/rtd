import socket
import pickle
import struct
import time
import numpy as np
import sys
import lunar_tools as lt
from rtd.utils.prompt_provider import PromptProviderMicrophone, PromptProviderTxtFile
from rtd.utils.audio_detector import AudioDetector
from rtd.utils.oscillators import Oscillator

if len(sys.argv) > 1 and sys.argv[1].lower() == "server":
    from rtd.sdxl_turbo.diffusion_engine import DiffusionEngine
    from rtd.sdxl_turbo.embeddings_mixer import EmbeddingsMixer
    from rtd.dynamic_processor.processor_dynamic_module import DynamicProcessor
    from rtd.utils.input_image import InputImageProcessor, AcidProcessor
    from rtd.utils.optical_flow import OpticalFlowEstimator
    from rtd.utils.posteffect import Posteffect

from rtd.utils.input_image import InputImageProcessor, AcidProcessor
from rtd.utils.compression_helpers import send_compressed, recv_compressed

###############################################################################
# SubmersionServer
###############################################################################
class SubmersionServer:
    def __init__(self, host="0.0.0.0", port=9999, device="cuda:0", do_diffusion=True, do_compile=True, bounce=False):
        self.host = host
        self.port = port
        self.device = device
        self.do_diffusion = do_diffusion
        self.do_compile = do_compile
        self.bounce = bounce  # If True, the server will simply echo back the received image without processing

        # These dimensions match the submersion pipeline settings.
        self.height_diffusion = 384 + 96
        self.width_diffusion = 512 + 128

        # Create and bind the server socket.
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm to reduce latency.
        self.server_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"SubmersionServer listening on {self.host}:{self.port}")

        if not self.bounce:
            # Initialize image processing modules.
            self.input_image_processor = InputImageProcessor(device=device)
            self.input_image_processor.set_flip(do_flip=True, flip_axis=1)

            self.acid_processor = AcidProcessor(
                height_diffusion=self.height_diffusion,
                width_diffusion=self.width_diffusion,
                device=device,
            )
            self.dynamic_processor = DynamicProcessor()
            
            # Initialize optical flow and posteffect
            self.opt_flow_estimator = OpticalFlowEstimator(use_ema=False)
            self.posteffect_processor = Posteffect()

            self.de_img = DiffusionEngine(
                use_image2image=True,
                height_diffusion_desired=self.height_diffusion,
                width_diffusion_desired=self.width_diffusion,
                do_compile=self.do_compile,
                do_diffusion=self.do_diffusion,
                device=device,
            )
            # Optionally, set a default prompt embedding.
            if self.do_diffusion:
                init_prompt = 'Bizarre creature from Hieronymus Bosch painting "A Garden of Earthly Delights" on a schizophrenic ayahuasca trip'
                em = EmbeddingsMixer(self.de_img.pipe)
                embeds = em.encode_prompt(init_prompt)
                self.de_img.set_embeddings(embeds)

            # Store the last generated diffusion image (used by dynamic_processor).
            self.last_diffused = None
        else:
            print("Bounce mode enabled: Server will echo the received image without processing.")

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes from the socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self, sock):
        """Receive a length-prefixed message."""
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        return self.recvall(sock, msglen)

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack('!I', len(msg)) + msg
        sock.sendall(msg)

    def handle_client(self, client_sock, addr):
        print(f"Connected by {addr}")
        while True:
            # try:
            data = self.recv_msg(client_sock)
            if data is None:
                print("Client disconnected")
                break

            # Unpickle the received payload (contains processing parameters).
            payload = pickle.loads(data)

            if not self.bounce:
                mic_prompt = payload.get("mic_prompt")
                if mic_prompt:
                    print(f"New microphone prompt received: {mic_prompt}")
                    if self.do_diffusion:
                        em = EmbeddingsMixer(self.de_img.pipe)
                        embeds = em.encode_prompt(mic_prompt)
                        self.de_img.set_embeddings(embeds)
                
                txt_file_prompt = payload.get("txt_file_prompt")
                if txt_file_prompt:
                    print(f"New text file prompt received: {txt_file_prompt}")
                    if self.do_diffusion:
                        em = EmbeddingsMixer(self.de_img.pipe)
                        embeds = em.encode_prompt(txt_file_prompt)
                        self.de_img.set_embeddings(embeds)

                dynamic_transcript = payload.get("dynamic_transcript")
                if dynamic_transcript:
                    print(f"New dynamic transcript received: {dynamic_transcript}")
                    self.dynamic_processor.update_protoblock(dynamic_transcript)
                    
                # Check for dynamic processor control signals
                dyn_prompt_restore_backup = payload.get("dyn_prompt_restore_backup", False)
                if dyn_prompt_restore_backup:
                    self.dynamic_processor.restore_backup()
                    
                dyn_prompt_del_current = payload.get("dyn_prompt_del_current", False)
                if dyn_prompt_del_current:
                    self.dynamic_processor.delete_current_fn_func()

            if self.bounce:
                # In bounce mode, receive the compressed image and echo it back.
                img = recv_compressed(client_sock)
                if img is None:
                    break
                send_compressed(client_sock, img, quality=90)
                continue

            # Receive the compressed camera image.
            img_cam = recv_compressed(client_sock)
            if img_cam is None or not isinstance(img_cam, np.ndarray):
                print("Invalid image received")
                continue

            # Extract processing parameters.
            do_human_seg = payload.get("do_human_seg", True)
            acid_strength = payload.get("acid_strength", 0.11)
            acid_strength_foreground = payload.get("acid_strength_foreground", 0.11)
            coef_noise = payload.get("coef_noise", 0.15)
            zoom_factor = payload.get("zoom_factor", 1.0)
            x_shift = payload.get("x_shift", 0)
            y_shift = payload.get("y_shift", 0)
            color_matching = payload.get("color_matching", 0.5)
            dynamic_func_coef1 = payload.get("dynamic_func_coef1", 0.5)
            dynamic_func_coef2 = payload.get("dynamic_func_coef2", 0.5)
            dynamic_func_coef3 = payload.get("dynamic_func_coef3", 0.5)
            do_dynamic_processor = payload.get("do_dynamic_processor", False)
            do_blur = payload.get("do_blur", False)
            do_acid_tracers = payload.get("do_acid_tracers", True)
            do_acid_wobblers = payload.get("do_acid_wobblers", False)
            brightness = payload.get("brightness", 1.0)
            do_infrared_colorize = payload.get("do_infrared_colorize", False)
            
            # Postprocessing parameters
            do_postproc = payload.get("do_postproc", True)
            postproc_func_coef1 = payload.get("postproc_func_coef1", 0.5)
            postproc_func_coef2 = payload.get("postproc_func_coef2", 0.5)
            postproc_mod_button1 = payload.get("postproc_mod_button1", True)
            sound_volume = payload.get("sound_volume", 0)

            # Process the received image using InputImageProcessor.
            self.input_image_processor.set_human_seg(do_human_seg)
            self.input_image_processor.set_resizing_factor_humanseg(0.4)
            self.input_image_processor.set_blur(do_blur)
            self.input_image_processor.set_brightness(brightness)
            self.input_image_processor.set_infrared_colorize(do_infrared_colorize)
            img_proc, human_seg_mask = self.input_image_processor.process(img_cam)
            
            if not do_human_seg:
                human_seg_mask = np.ones_like(img_proc).astype(np.float32) / 255
            
            # Calculate optical flow for posteffect processing
            # Ensure consistent dimensions for optical flow processing
            if not hasattr(self, 'last_img_for_flow') or self.last_img_for_flow is None:
                self.last_img_for_flow = img_cam.copy()
            
            # Ensure both images have the same dimensions
            if self.last_img_for_flow.shape != img_cam.shape:
                # Resize the current image to match the dimensions of the last image
                import cv2
                img_cam_resized = cv2.resize(img_cam, (self.last_img_for_flow.shape[1], self.last_img_for_flow.shape[0]))
                opt_flow = self.opt_flow_estimator.get_optflow(img_cam_resized, 
                                                         low_pass_kernel_size=55, window_length=55)
            else:
                opt_flow = self.opt_flow_estimator.get_optflow(img_cam.copy(), 
                                                         low_pass_kernel_size=55, window_length=55)
            
            # Store current image for next iteration
            self.last_img_for_flow = img_cam.copy()

            if do_dynamic_processor and self.last_diffused is not None:
                # If dynamic processing is enabled and a previous diffusion exists,
                # run the dynamic processor.
                img_acid = self.dynamic_processor.process(
                    np.flip(img_proc, axis=1).astype(np.float32),
                    human_seg_mask.astype(np.float32) / 255,
                    np.flip(self.last_diffused.astype(np.float32), axis=1).copy(),
                    opt_flow,
                    dynamic_func_coef1,
                )
                img_acid = np.clip(img_acid, 0, 255).astype(np.uint8)
            else:
                # Else, use the acid processor.
                self.acid_processor.set_acid_strength(acid_strength)
                self.acid_processor.set_coef_noise(coef_noise)
                self.acid_processor.set_acid_tracers(do_acid_tracers)
                self.acid_processor.set_acid_strength_foreground(acid_strength_foreground)
                self.acid_processor.set_zoom_factor(zoom_factor)
                self.acid_processor.set_x_shift(x_shift)
                self.acid_processor.set_y_shift(y_shift)
                self.acid_processor.set_do_acid_wobblers(do_acid_wobblers)
                self.acid_processor.set_color_matching(color_matching)
                img_acid = self.acid_processor.process(img_proc, human_seg_mask)

            # Diffusion processing.
            self.de_img.set_input_image(img_acid)
            self.de_img.set_guidance_scale(0.5)
            self.de_img.set_strength(1 / self.de_img.num_inference_steps + 0.00001)
            img_diffusion = np.array(self.de_img.generate())
            
            # Apply posteffect processing if enabled
            if do_postproc and opt_flow is not None:
                output_to_render, update_img = self.posteffect_processor.process(
                    img_diffusion, 
                    human_seg_mask.astype(np.float32) / 255, 
                    opt_flow,
                    postproc_func_coef1,
                    postproc_func_coef2,
                    postproc_mod_button1,
                    sound_volume
                )
            else:
                output_to_render = img_diffusion
                update_img = img_diffusion

            # Update acid_processor and store the latest diffusion for potential dynamic processing.
            self.acid_processor.update(update_img)
            self.last_diffused = img_diffusion

            # Send the rendered image back to the client using compression.
            send_compressed(client_sock, output_to_render, quality=90)

            # If debug mode is requested, also send the input processed image
            do_debug_seethrough = payload.get("do_debug_seethrough", False)
            if do_debug_seethrough:
                send_compressed(client_sock, img_proc, quality=90)
            else:
                # Send a placeholder to maintain protocol consistency
                send_compressed(client_sock, np.zeros((1, 1, 3), dtype=np.uint8), quality=90)

            # except Exception as e:
            #     print("Error handling client:", e)
            #     break

        client_sock.close()

    def serve_forever(self):
        """Main loop to accept and serve clients."""
        while True:
            client_sock, addr = self.server_socket.accept()
            self.handle_client(client_sock, addr)

###############################################################################
# SubmersionClient
###############################################################################
class SubmersionClient:
    def __init__(self, server_host="localhost", server_port=9999):
        self.server_host = server_host
        self.server_port = server_port

        # Camera settings.
        self.shape_hw_cam = (576, 1024)
        # Renderer settings.
        self.width_render = 1920
        self.height_render = 1080
        self.do_fullscreen = True

        # Initialize LT camera, meta input, and renderer.
        self.cam = lt.WebCam(shape_hw=self.shape_hw_cam)
        self.meta_input = lt.MetaInput()
        self.renderer = lt.Renderer(
            width=self.width_render,
            height=self.height_render,
            backend="opencv",
            do_fullscreen=self.do_fullscreen,
        )

        # Initialize the microphone prompt and speech detection:
        self.speech_detector = lt.Speech2Text()
        self.prompt_provider_microphone = PromptProviderMicrophone()
        
        # Initialize text file prompt provider
        self.prompt_provider_txt_file = PromptProviderTxtFile('materials/prompts/dancing_fibers.txt')
        
        # Initialize audio detector and oscillator
        self.audio_detector = AudioDetector()
        self.oscillator = Oscillator()
        
        # Initialize FPS tracking
        self.fps_tracker = lt.FPSTracker()

        # Connect to the server.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Disable Nagle's algorithm for lower latency.
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.connect((self.server_host, self.server_port))
        print(f"Connected to server at {self.server_host}:{self.server_port}")

    def send_msg(self, sock, msg):
        """Send a length-prefixed message."""
        msg = struct.pack('!I', len(msg)) + msg
        sock.sendall(msg)

    def recvall(self, sock, n):
        """Helper: receive exactly n bytes."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def recv_msg(self, sock):
        """Receive a length-prefixed message."""
        raw_msglen = self.recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('!I', raw_msglen)[0]
        return self.recvall(sock, msglen)
    
    def run(self):
        while True:
            self.fps_tracker.start_segment("Client Processing")
            t_processing_start = time.time()

            # Acquire meta-input parameters (matching those used in the server).
            # Boolean parameters
            new_prompt_mic_unmuter = self.meta_input.get(akai_lpd8="A1", akai_midimix="A3", button_mode="held_down")
            do_cycle_prompt_from_file = self.meta_input.get(akai_lpd8="C0", akai_midimix="A4", button_mode="pressed_once")
            dyn_prompt_mic_unmuter = self.meta_input.get(akai_lpd8="A0", akai_midimix="B3", button_mode="held_down")
            do_dynamic_processor = self.meta_input.get(akai_lpd8="B0", akai_midimix="B4", button_mode="toggle", val_default=False)
            do_human_seg = self.meta_input.get(akai_lpd8="B1", akai_midimix="E3", button_mode="toggle", val_default=True)
            do_acid_wobblers = self.meta_input.get(akai_lpd8="C1", akai_midimix="D3", button_mode="toggle", val_default=False)
            do_infrared_colorize = self.meta_input.get(akai_lpd8="D0", akai_midimix="H4", button_mode="toggle", val_default=False)
            do_debug_seethrough = self.meta_input.get(akai_lpd8="D1", akai_midimix="H3", button_mode="toggle", val_default=False)
            do_postproc = self.meta_input.get(akai_midimix="G3", button_mode="toggle", val_default=True)
            do_audio_modulation = self.meta_input.get(akai_midimix="D4", button_mode="toggle", val_default=False)
            do_param_oscillators = self.meta_input.get(akai_midimix="C3", button_mode="toggle", val_default=False)
            
            dyn_prompt_restore_backup = self.meta_input.get(akai_midimix="F3", button_mode="released_once")
            dyn_prompt_del_current = self.meta_input.get(akai_midimix="F4", button_mode="released_once")

            # Floating point parameters
            acid_strength = self.meta_input.get(akai_lpd8="E0", akai_midimix="C0", val_min=0, val_max=1.0, val_default=0.05)
            acid_strength_foreground = self.meta_input.get(akai_lpd8="E1", akai_midimix="C1", val_min=0, val_max=1.0, val_default=0.05)
            coef_noise = self.meta_input.get(akai_lpd8="F0", akai_midimix="C2", val_min=0, val_max=0.3, val_default=0.05)
            zoom_factor = self.meta_input.get(akai_lpd8="F1", akai_midimix="H2", val_min=0.5, val_max=1.5, val_default=1.0)
            x_shift = int(self.meta_input.get(akai_midimix="H0", val_min=-50, val_max=50, val_default=0))
            y_shift = int(self.meta_input.get(akai_midimix="H1", val_min=-50, val_max=50, val_default=0))
            color_matching = self.meta_input.get(akai_lpd8="G0", akai_midimix="G0", val_min=0, val_max=1, val_default=0.5)
            brightness = self.meta_input.get(akai_midimix="A2", val_min=0.0, val_max=2, val_default=1.0)

            dynamic_func_coef1 = self.meta_input.get(akai_midimix="F0", val_min=0, val_max=1, val_default=0.5)
            dynamic_func_coef2 = self.meta_input.get(akai_midimix="F1", val_min=0, val_max=1, val_default=0.5)
            dynamic_func_coef3 = self.meta_input.get(akai_midimix="F2", val_min=0, val_max=1, val_default=0.5)

            # Postprocessing parameters
            postproc_func_coef1 = self.meta_input.get(akai_lpd8="H0", akai_midimix="G1", val_min=0, val_max=1, val_default=0.5)
            postproc_func_coef2 = self.meta_input.get(akai_lpd8="H1", akai_midimix="G2", val_min=0, val_max=1, val_default=0.5)
            postproc_mod_button1 = self.meta_input.get(akai_midimix="G4", button_mode="toggle", val_default=True)
            
            # Oscillator-based control
            if do_param_oscillators:
                do_cycle_prompt_from_file = self.oscillator.get('prompt_cycle', 60, 0, 1, 'trigger')
                acid_strength = self.oscillator.get('acid_strength', 30, 0, 0.5, 'continuous')
                coef_noise = self.oscillator.get('coef_noise', 2, 60, 0.15, 'continuous')
                postproc_func_coef1 = self.oscillator.get('postproc_func_coef1', 120, 0.25, 1, 'continuous')
                postproc_func_coef2 = self.oscillator.get('postproc_func_coef2', 180, 0, 0.5, 'continuous')

            # Sound-based control
            sound_volume = 0
            if do_audio_modulation:
                sound_volume = self.audio_detector.get_last_volume()
                
            # Fixed parameters
            do_blur = False
            do_acid_tracers = True

            # Process microphone inputs for prompt updates
            new_diffusion_prompt_available_from_mic = self.prompt_provider_microphone.handle_unmute_button(new_prompt_mic_unmuter)
            if new_diffusion_prompt_available_from_mic:
                mic_prompt = self.prompt_provider_microphone.get_current_prompt()
                print(f"Client new prompt: {mic_prompt}")
            else:
                mic_prompt = None
                
            # Handle prompt cycling from text file
            txt_file_prompt = None
            if do_cycle_prompt_from_file:
                self.prompt_provider_txt_file.handle_prompt_cycling_button(do_cycle_prompt_from_file)
                txt_file_prompt = self.prompt_provider_txt_file.get_current_prompt()
                print(f"Client new text file prompt: {txt_file_prompt}")

            # Process dynamic speech input
            new_dynamic_prompt_available = self.speech_detector.handle_unmute_button(dyn_prompt_mic_unmuter)
            if new_dynamic_prompt_available:
                dynamic_transcript = self.speech_detector.transcript
            else:
                dynamic_transcript = None

            # Acquire the camera image.
            self.fps_tracker.start_segment("Camera Capture")
            img_cam = self.cam.get_img()

            # Prepare the payload with processing parameters (exclude raw image).
            payload = {
                "do_human_seg": do_human_seg,
                "acid_strength": acid_strength,
                "acid_strength_foreground": acid_strength_foreground,
                "coef_noise": coef_noise,
                "zoom_factor": zoom_factor,
                "x_shift": x_shift,
                "y_shift": y_shift,
                "color_matching": color_matching,
                "dynamic_func_coef1": dynamic_func_coef1,
                "dynamic_func_coef2": dynamic_func_coef2,
                "dynamic_func_coef3": dynamic_func_coef3,
                "do_dynamic_processor": do_dynamic_processor,
                "do_blur": do_blur,
                "do_acid_tracers": do_acid_tracers,
                "do_acid_wobblers": do_acid_wobblers,
                "mic_prompt": mic_prompt,
                "txt_file_prompt": txt_file_prompt,
                "dynamic_transcript": dynamic_transcript,
                "brightness": brightness,
                "do_infrared_colorize": do_infrared_colorize,
                "do_debug_seethrough": do_debug_seethrough,
                "do_postproc": do_postproc,
                "postproc_func_coef1": postproc_func_coef1,
                "postproc_func_coef2": postproc_func_coef2,
                "postproc_mod_button1": postproc_mod_button1,
                "sound_volume": sound_volume,
                "dyn_prompt_restore_backup": dyn_prompt_restore_backup,
                "dyn_prompt_del_current": dyn_prompt_del_current,
            }

            try:
                self.fps_tracker.start_segment("Network Communication")
                # Use highest protocol for faster serialization for parameters.
                data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
                self.send_msg(self.sock, data)
                # Send the camera image using compression.
                send_compressed(self.sock, img_cam, quality=90)

                # Wait for the processed diffusion image (received in compressed form).
                img_diffusion = recv_compressed(self.sock)
                if img_diffusion is None:
                    print("Disconnected from server")
                    break
                
                # Wait for debug image if requested
                debug_img = recv_compressed(self.sock)
                if debug_img is None:
                    print("Disconnected from server")
                    break

                self.fps_tracker.start_segment("Rendering")
                # Render the appropriate image based on debug mode
                if do_debug_seethrough and debug_img.size > 3:  # Check if it's not the placeholder
                    self.renderer.render(debug_img)
                else:
                    self.renderer.render(img_diffusion)

                t_processing = time.time() - t_processing_start
                # Update and display FPS (this will handle the last segment timing)
                self.fps_tracker.print_fps()
            except Exception as e:
                print("Error during communication with server:", e)
                break

        self.sock.close()

###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [server|client]")
        sys.exit(1)

    role = sys.argv[1].lower()
    if len(sys.argv) > 2:
        server_ip = sys.argv[2].lower()
    else:
        server_ip = "localhost"

    if role == "server":
        # To test pure network latency, enable bounce mode by setting bounce=True.
        server = SubmersionServer(bounce=False)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server shutting down.")
    elif role == "client":
        client = SubmersionClient(server_host=server_ip)
        client.run()
    else:
        print("Invalid mode. Use 'server' or 'client'.")
