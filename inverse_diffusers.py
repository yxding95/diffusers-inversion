import torch
import argparse
from PIL import Image
from inverse_pipeline import InversePipeline
from inverse_pipeline_xl import InversePipelineXL
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, AutoencoderKL
from transformers import AutoTokenizer
from schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from clip import ExceptionCLIPTextModel, ExceptionCLIPTextModelWithProj
import numpy as np
    
def inverse_SD2_1_512(input_image_path, model_path="stabilityai/stable-diffusion-2-1-base", inf_steps=20, outfolder="./outputs/", vis=True, device="cuda:0"):
    # default DPMSolver, with exception
    prefix = f"{input_image_path.split('/')[-1].split('.')[0]}_SD2_1_512"
    
    exclip = ExceptionCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    pipe = InversePipeline.from_pretrained(model_path, text_encoder=exclip).to(device)
    pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    image = Image.open(input_image_path).resize((512,512), Image.Resampling.LANCZOS).convert("RGB")
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=inf_steps,
        image=image,
    )
    
    noise_image, noise, decode_image, intermediate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["intermediate_latents"][0]
    print("Noise mean: ", noise.mean(), "Noise std: ", noise.std())
    
    del pipe
    torch.cuda.empty_cache()
    
    if vis:
        denoise_pipe = StableDiffusionPipeline.from_pretrained(model_path, text_encoder=exclip).to(device)
        denoise_pipe.scheduler = DPMSolverMultistepScheduler.from_config(denoise_pipe.scheduler.config)
        outputs = denoise_pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=inf_steps,
            latents=noise.unsqueeze(0)
        ) 
        recon_image = outputs["images"][0]
        
        noise_image.save(outfolder + f"{prefix}_noise.jpg")
        decode_image.save(outfolder + f"{prefix}_decode.jpg")
        recon_image.save(outfolder + f"{prefix}_recon.jpg")
        
    return noise, intermediate_latents


def inverse_SD2_1_768(input_image_path, model_path="stabilityai/stable-diffusion-2-1", inf_steps=20, outfolder="./outputs/", vis=True, device="cuda:0"):
    # default DPMSolver, with exception
    prefix = f"{input_image_path.split('/')[-1].split('.')[0]}_SD2_1_768"
    
    exclip = ExceptionCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    pipe = InversePipeline.from_pretrained(model_path, text_encoder=exclip).to(device)
    pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    image = Image.open(input_image_path).resize((768,768), Image.Resampling.LANCZOS).convert("RGB")
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=inf_steps,
        image=image,
    )
    
    noise_image, noise, decode_image, intermediate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["intermediate_latents"][0]
    print("Noise mean: ", noise.mean(), "Noise std: ", noise.std())
    
    del pipe
    torch.cuda.empty_cache()
    
    if vis:
        denoise_pipe = StableDiffusionPipeline.from_pretrained(model_path, text_encoder=exclip).to(device)
        denoise_pipe.scheduler = DPMSolverMultistepScheduler.from_config(denoise_pipe.scheduler.config)
        outputs = denoise_pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=inf_steps,
            latents=noise.unsqueeze(0)
        ) 
        recon_image = outputs["images"][0]
        
        noise_image.save(outfolder + f"{prefix}_noise.jpg")
        decode_image.save(outfolder + f"{prefix}_decode.jpg")
        recon_image.save(outfolder + f"{prefix}_recon.jpg")
        
    return noise, intermediate_latents


def inverse_SDXL_1024(input_image_path, model_path="stabilityai/stable-diffusion-xl-base-1.0", inf_steps=50, outfolder="./outputs/", vis=True, device="cuda:0"):
    # DPMSolverInvert
    prefix = f"{input_image_path.split('/')[-1].split('.')[0]}_SDXL_1024"
    
    exclip = ExceptionCLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device)
    exclip_2 = ExceptionCLIPTextModelWithProj.from_pretrained(model_path, subfolder="text_encoder_2").to(device)
    # vae = AutoencoderKL.from_pretrained("HeWhoRemixes/pastelmix-better-vae-fp32", subfolder="vae").to(device)
    pipe = InversePipelineXL.from_pretrained(model_path, text_encoder = exclip, text_encoder_2=exclip_2).to(device)
    pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    image = Image.open(input_image_path).resize((1024,1024), Image.Resampling.LANCZOS).convert('RGB')
    
    prompt_str = ""
    outputs = pipe(
        prompt_str, 
        guidance_scale=1,
        num_inference_steps=inf_steps,
        image = image
    )

    noise_image, noise, decode_image, intermidiate_latents = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0], outputs["intermediate_latents"][0]
    print("Noise mean: ", noise.mean(), "Noise std: ", noise.std())
    
    del pipe
    torch.cuda.empty_cache()
    
    if vis:
        denoise_pipe = StableDiffusionXLPipeline.from_pretrained(model_path, text_encoder=exclip, text_encoder_2=exclip_2).to(device)
        denoise_pipe.scheduler = DPMSolverMultistepScheduler.from_config(denoise_pipe.scheduler.config)
        outputs = denoise_pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=inf_steps,
            latents=noise.unsqueeze(0)
        ) 
        recon_image = outputs["images"][0]
        
        noise_image.save(outfolder + f"{prefix}_noise.jpg")
        decode_image.save(outfolder + f"{prefix}_decode.jpg")
        recon_image.save(outfolder + f"{prefix}_recon.jpg")
    
    return noise, intermidiate_latents

def args_parse():
    pass

if __name__ == "__main__":
    final_inverted_latent, intermidiate_latents = inverse_SD2_1_512("/home/zicheng/Projects/diffusers-inversion/assets/test_images/dalle3_1024.png")
    final_inverted_latent, intermidiate_latents = inverse_SD2_1_768("/home/zicheng/Projects/diffusers-inversion/assets/test_images/dalle3_1024.png")
    final_inverted_latent, intermidiate_latents = inverse_SDXL_1024("/home/zicheng/Projects/diffusers-inversion/assets/test_images/dalle3_1024.png",  inf_steps=20, vis=True)