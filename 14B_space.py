#!/usr/bin/env python3
"""
Script to generate images and videos using text2image and video2world pipelines.
Reads prompts from prompts/space/space_img.txt and prompts/space/space_vid.txt
Supports configurable model sizes and output paths.
"""

import argparse
import os
from pathlib import Path

# Import the functions we need from both examples
from examples.text2image import setup_pipeline as setup_text2image_pipeline, generate_image, cleanup_distributed
from examples.video2world import setup_pipeline as setup_video2world_pipeline, generate_video


def read_prompt_from_file(prompt_file_path: str) -> str:
    """Read prompt from text file, handling line continuations with backslashes."""
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Handle line continuations with backslashes
    # Remove backslash at end of lines and join lines
    lines = content.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if line.endswith('\\'):
            # Remove backslash and add space
            processed_lines.append(line[:-1].strip() + ' ')
        else:
            processed_lines.append(line)
    
    # Join all lines into a single prompt
    prompt = ''.join(processed_lines).strip()
    return prompt


def create_text2image_args(prompt: str, output_path: str, model_size: str) -> argparse.Namespace:
    """Create arguments object configured for text2image generation."""
    args = argparse.Namespace()
    
    # Model configuration
    args.model_size = model_size  # Use specified model size
    args.distill_steps = 0   # Use original non-distilled model
    args.dit_path = ""       # Use default checkpoint
    args.load_ema = False    # Don't use EMA weights
    
    # Generation parameters
    args.prompt = prompt
    args.negative_prompt = ""
    args.aspect_ratio = "16:9"
    args.seed = 42  # Default seed (will be overridden)
    
    # Output settings
    args.save_path = output_path
    args.batch_input_json = None
    
    # Performance settings
    args.use_cuda_graphs = False
    args.benchmark = False
    args.use_fast_tokenizer = False
    
    # Guardrail settings
    args.disable_guardrail = False
    args.offload_guardrail = True  # Offload to CPU to save GPU memory
    
    return args


def create_video2world_args(prompt: str, input_image_path: str, output_video_path: str, model_size: str) -> argparse.Namespace:
    """Create arguments object configured for video2world generation."""
    args = argparse.Namespace()
    
    # Model configuration
    args.model_size = model_size  # Use specified model size
    args.resolution = "720"  # Use 720p resolution
    args.fps = 16           # Use 16 FPS
    args.dit_path = ""      # Use default checkpoint
    args.load_ema = False   # Don't use EMA weights
    
    # Generation parameters
    args.prompt = prompt
    args.input_path = input_image_path
    args.negative_prompt = "The video captures a series of frames showing ugly scenes,\
                            static with no motion, motion blur, over-saturation, shaky footage, low resolution,\
                            grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes,\
                            poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate,\
                            artifacting, color banding, unnatural transitions, outdated special effects, fake elements,\
                            unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall,\
                            the video is of poor quality."
    args.aspect_ratio = "16:9"
    args.num_conditional_frames = 1  # Single frame conditioning (from image)
    args.guidance = 7.0
    args.seed = 42  # Default seed (will be overridden)
    
    # Output settings
    args.save_path = output_video_path
    args.batch_input_json = None
    
    # Multi-GPU settings
    args.num_gpus = 1  # Single GPU for now
    
    # Performance settings
    args.benchmark = False
    args.use_cuda_graphs = False
    args.natten = False
    
    # Guardrail and prompt refiner settings
    args.disable_guardrail = False
    args.offload_guardrail = True
    args.disable_prompt_refiner = False
    args.offload_prompt_refiner = True
    args.offload_text_encoder = True
    args.downcast_text_encoder = True
    
    return args


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate space images and videos using Cosmos Predict2 pipelines"
    )
    parser.add_argument(
        "--model_size",
        choices=["2B", "5B", "14B"],
        default="14B",
        help="Size of the model to use for generation (default: 14B)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/space_text2img_img2video",
        help="Output directory for generated files (default: output/space_text2img_img2video)"
    )
    parser.add_argument(
        "--image_prompt",
        type=str,
        default="prompts/space/space_img.txt",
        help="Path to text file containing image generation prompt (default: prompts/space/space_img.txt)"
    )
    parser.add_argument(
        "--video_prompt",
        type=str,
        default="prompts/space/space_vid.txt", 
        help="Path to text file containing video generation prompt (default: prompts/space/space_vid.txt)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)"
    )
    return parser.parse_args()


def main():
    """Main function to generate image and then video from space prompts."""
    
    # Parse command line arguments
    cmd_args = parse_args()
    
    # Paths
    workspace_root = Path(__file__).parent
    img_prompt_file = workspace_root / cmd_args.image_prompt
    vid_prompt_file = workspace_root / cmd_args.video_prompt
    
    # Output directory and files
    output_dir = Path(cmd_args.output_dir)
    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_image_path = output_dir / "space_image.jpg"
    output_video_path = output_dir / "space_video.mp4"
    
    print(f"Model size: {cmd_args.model_size}")
    print(f"Reading image prompt from: {img_prompt_file}")
    print(f"Reading video prompt from: {vid_prompt_file}")
    print(f"Output directory: {output_dir}")
    print(f"Image will be saved to: {output_image_path}")
    print(f"Video will be saved to: {output_video_path}")
    print(f"Random seed: {cmd_args.seed}")
    
    try:
        # Step 1: Generate image using text2image
        print("\n" + "="*60)
        print("STEP 1: GENERATING IMAGE WITH TEXT2IMAGE PIPELINE")
        print("="*60)
        
        # Read the image prompt
        img_prompt = read_prompt_from_file(str(img_prompt_file))
        print(f"Image prompt: {img_prompt[:100]}...")  # Show first 100 chars
        
        # Create arguments for text2image
        img_args = create_text2image_args(img_prompt, str(output_image_path), cmd_args.model_size)
        img_args.seed = cmd_args.seed  # Use command line seed
        
        print(f"Setting up {cmd_args.model_size} Text2Image pipeline...")
        # Setup the text2image pipeline
        text2img_pipe = setup_text2image_pipeline(img_args)
        
        if text2img_pipe is not None:
            print("Generating image...")
            # Generate the image
            generate_image(img_args, text2img_pipe)
            print("Image generation completed!")
            
            # Clean up text2image pipeline
            del text2img_pipe
            cleanup_distributed()
        else:
            print("Text2Image pipeline setup failed - pipe is None")
            return
        
        # Step 2: Generate video using video2world
        print("\n" + "="*60)
        print("STEP 2: GENERATING VIDEO WITH VIDEO2WORLD PIPELINE")
        print("="*60)
        
        # Read the video prompt
        vid_prompt = read_prompt_from_file(str(vid_prompt_file))
        print(f"Video prompt: {vid_prompt[:100]}...")  # Show first 100 chars
        
        # Create arguments for video2world
        vid_args = create_video2world_args(vid_prompt, str(output_image_path), str(output_video_path), cmd_args.model_size)
        vid_args.seed = cmd_args.seed  # Use command line seed
        
        print(f"Setting up {cmd_args.model_size} Video2World pipeline...")
        # Setup the video2world pipeline
        video2world_pipe = setup_video2world_pipeline(vid_args)
        
        if video2world_pipe is not None:
            print("Generating video from image...")
            # Generate the video
            generate_video(vid_args, video2world_pipe)
            print("Video generation completed!")
            
            print(f"\n🎉 SUCCESS! Generated files:")
            print(f"   📸 Image: {output_image_path}")
            print(f"   🎬 Video: {output_video_path}")
        else:
            print("Video2World pipeline setup failed - pipe is None")
            
    except Exception as e:
        print(f"Error during generation: {e}")
        raise
    finally:
        # Clean up distributed environment
        cleanup_distributed()


if __name__ == "__main__":
    main() 