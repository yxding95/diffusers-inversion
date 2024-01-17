import torch
import argparse
from PIL import Image
from inverse_pipeline import InversePipeline
from inverse_pipeline_xl import InversePipelineXL
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
from transformers import AutoTokenizer
from schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from clip import ExceptionCLIPTextModel, ExceptionCLIPTextModelWithProj
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='assets/test_images/spiderman.png')
    parser.add_argument('--results_folder', type=str, default='outputs/')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--model_path', type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    args = parser.parse_args()

    exclip = ExceptionCLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    exclip_2 = ExceptionCLIPTextModelWithProj.from_pretrained(args.model_path, subfolder="text_encoder_2").to(device)
    pipe = InversePipelineXL.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    # pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    # pipe.scheduler = InverseDDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = InversePNDMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    image = Image.open(args.input_image).resize((1024,1024), Image.Resampling.LANCZOS).convert("RGB")
    # x0 = np.array(image)/255
    # x0 = torch.from_numpy(x0).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
    # x0 = (x0 - 0.5) * 2.
    # with torch.no_grad():
    #     img_latents = pipe.vae.encode(x0.float()).latent_dist.sample().to(device)
    #     img_latents *= pipe.vae.config.scaling_factor
    
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        output_type="latent",
        image = image
    )

    noisy_latent = outputs["images"][0]
    print(noisy_latent.mean(), noisy_latent.std())
    
    del pipe
    
    denoise_pipe = StableDiffusionXLPipeline.from_pretrained(args.model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    denoise_pipe.scheduler = DPMSolverMultistepScheduler.from_config(denoise_pipe.scheduler.config)
    outputs = denoise_pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=args.num_inference_steps,
        latents=noisy_latent.unsqueeze(0)
    ) 
    recon_image = outputs["images"][0]
    recon_image.save(args.results_folder + "recon.jpg")




