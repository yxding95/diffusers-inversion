import torch
import argparse
from glob import glob
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image
from inverse_pipeline import InversePipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler, PNDMScheduler, EulerDiscreteScheduler, DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler
from schedulers import InverseDDIMScheduler, InversePNDMScheduler, InverseEulerDiscreteScheduler
from clip import ExceptionCLIPTextModel
from datasets import load_dataset
from torchvision import transforms

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='../minecraft-preview')
    parser.add_argument('--results_folder', type=str, default='../minecraft-preview-inversion-all-lora-text/')
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--model_path', type=str, default="../../stable-diffusion-2-1-base/")
    args = parser.parse_args()

    os.makedirs(args.results_folder, exist_ok=True)

    exclip = ExceptionCLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder").to(device)
    #pipe = InversePipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
    pipe = InversePipeline.from_pretrained(args.model_path).to(device)
    pipe.load_lora_weights("../sdtb/sd-minecraft-model-all-lora/")
    #pipe = StableDiffusionPipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
    pipe.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)

    dataset = load_dataset(
        args.dataset_name,
    )

    all_image_name = [args.results_folder + f"{i}.pt" for i, _ in enumerate(dataset.data["train"].to_pydict()["image"])]
    #print(all_image_name)
    dataset["train"] = dataset["train"].add_column("noise", all_image_name)
    column_names = dataset["train"].column_names

    image_column = "image"
    caption_column = "text"
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = pipe.tokenizer(
            captions, max_length=pipe.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids, captions

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            #transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
            #transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"], examples["raw_captions"] = tokenize_captions(examples)
        #examples["noises"] = [_ for _ in examples["noise"]]
        return examples
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        noises = [example["noise"] for example in examples]
        raw_captions = [example["raw_captions"] for example in examples]
        return {"pixel_values": pixel_values, "input_ids": input_ids, "raw_captions": raw_captions, "noises":noises}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=4,
    )

    for step, batch in enumerate(train_dataloader):
        model_input = pipe.vae.encode(batch["pixel_values"].to(pipe.device)).latent_dist.sample()
        model_input = model_input * pipe.vae.config.scaling_factor

        prompt_str = batch["raw_captions"][0]
        outputs = pipe(
            prompt_str, 
            #guidance_scale=1,
            num_inference_steps=args.num_inference_steps,
            latents=model_input,
            #image=image,
        )
        noise = outputs["noise"][0]
        #print(noise.shape)
        #print(batch["noises"])
        torch.save(noise, batch["noises"][0])

    '''
    if os.path.isdir(args.input_image):
        l_img_paths = sorted(glob(os.path.join(args.input_image, "*.png")))
    else:
        l_img_paths = [args.input_image]

    for img_path in tqdm(l_img_paths):
        bname = os.path.basename(img_path).split(".")[0]
        image = Image.open(img_path).convert("RGB").resize((512,512), Image.Resampling.LANCZOS)
        prompt_str = ""
        outputs = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inference_steps=args.num_inference_steps,
            image=image,
        )
        noise = outputs["noise"][0]
        torch.save(noise, os.path.join(args.results_folder, f"{bname}.pt"))

        # noise_image, noise, decode_image = outputs["images"][0], outputs["noise"][0], outputs["decode_images"][0]
        # noise_image.save(args.results_folder + "noise.jpg")
        # decode_image.save(args.results_folder + "decode.jpg")
        
        # denoise_pipe = StableDiffusionPipeline.from_pretrained(args.model_path, text_encoder=exclip).to(device)
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # outputs = denoise_pipe(
        #     prompt_str, 
        #     guidance_scale=1,
        #     num_inference_steps=args.num_inference_steps,
        #     latents=noise.unsqueeze(0)
        # ) 
        # recon_image = outputs["images"][0]
        # recon_image.save(args.results_folder + "recon.jpg")

    '''


