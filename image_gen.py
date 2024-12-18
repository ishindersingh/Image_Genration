import torch
from diffusers import StableDiffusionPipeline
import os

# Check if GPU is available, else fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Stable Diffusion pipeline
def load_pipeline():
    print("Loading Stable Diffusion pipeline...")
    model_id = "CompVis/stable-diffusion-v1-4"  # Use other models like "stabilityai/stable-diffusion-2-1"
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipeline.to(device)
    return pipeline

# Generate image from a prompt
def generate_image(pipeline, prompt, output_dir="output", image_name="generated_image.png"):
    print(f"Generating image for prompt: '{prompt}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate image
    image = pipeline(prompt).images[0]

    # Save image locally
    image_path = os.path.join(output_dir, image_name)
    image.save(image_path)
    print(f"Image saved to {image_path}")
    return image_path

if __name__ == "__main__":
    # Ensure the runtime is set to GPU
    if device != "cuda":
        raise EnvironmentError("GPU not available. Please ensure the runtime is set to GPU.")

    # Load the pipeline
    pipeline = load_pipeline()

    # User input for the prompt
    prompt = input("Enter a prompt for image generation: ")
    
    # Generate and save image
    output_path = generate_image(pipeline, prompt)
    print(f"Image generation completed: {output_path}")
