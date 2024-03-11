import torch
import argparse
from PIL import Image
from inverse_pipeline import InversePipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler
from schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from clip import ExceptionCLIPTextModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='assets/test_images/spiderman.png')
    parser.add_argument('--results_folder', type=str, default='outputs/')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default="../../stable-diffusion-2-1-base/")
    #parser.add_argument('--model_path', type=str, default="../sd-turbo/")
    args = parser.parse_args()

    exclip = ExceptionCLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    pipe = InversePipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
    #pipe = StableDiffusionPipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
    pipe.scheduler = InverseEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    image = Image.open(args.input_image).resize((512,512), Image.Resampling.LANCZOS)
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        image=image,
    )

    noise_image, noise, decode_image = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0]
    noise_image.save(args.results_folder + "noise.jpg")
    decode_image.save(args.results_folder + "decode.jpg")

    print(noise.mean(), noise.std())
    #noise = torch.randn_like(noise)
    #prompt_str = "spiderman"
    denoise_pipe = StableDiffusionPipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
    denoise_pipe.scheduler = EulerDiscreteScheduler.from_config(denoise_pipe.scheduler.config)
    outputs = denoise_pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        latents=noise.unsqueeze(0)
    ) 
    recon_image = outputs["images"][0]
    recon_image.save(args.results_folder + "recon.jpg")




