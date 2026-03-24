"""
DiT360 RunPod Serverless Handler
Generates 360° panoramic images from text prompts using DiT360.
"""

import runpod
import subprocess
import os
import base64


def handler(event):
    """
    Handler function for RunPod serverless.

    Input:
        - prompt (str): Text description for 360° panorama generation
        - seed (int): Random seed (default: 42)
        - num_inference_steps (int): Denoising steps (default: 28)
        - guidance_scale (float): Text guidance (default: 3.0)

    Returns:
        - image_base64 (str): Base64-encoded PNG image (1024x2048 equirectangular)
        - OR error message
    """
    try:
        job_input = event["input"]
        prompt = job_input.get("prompt", "A beautiful mountain landscape at sunset")
        seed = job_input.get("seed", 42)
        num_steps = job_input.get("num_inference_steps", 28)
        guidance_scale = job_input.get("guidance_scale", 3.0)

        output_dir = f"/tmp/dit360_output_{event['id']}"
        os.makedirs(output_dir, exist_ok=True)

        # Run DiT360 inference
        cmd = [
            "python", "inference.py",
            "--prompt", prompt,
            "--seed", str(seed),
            "--num_inference_steps", str(num_steps),
            "--guidance_scale", str(guidance_scale),
        ]

        result = subprocess.run(
            cmd,
            cwd="/app/DiT360",
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "OUTPUT_DIR": output_dir}
        )

        if result.returncode != 0:
            return {"error": f"Generation failed: {result.stderr[-500:]}"}

        # Find the output image
        output_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))]
        if not output_files:
            # Check DiT360 default output location
            dit360_dir = "/app/DiT360"
            output_files = [f for f in os.listdir(dit360_dir) if f.endswith(('.png', '.jpg')) and 'output' in f.lower()]
            if output_files:
                output_path = os.path.join(dit360_dir, output_files[0])
            else:
                return {"error": "No output image found", "stdout": result.stdout[-500:]}
        else:
            output_path = os.path.join(output_dir, output_files[0])

        # Read and encode the image
        with open(output_path, "rb") as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Cleanup
        os.remove(output_path)

        return {
            "image_base64": image_base64,
            "prompt": prompt,
            "format": "png",
            "resolution": "1024x2048"
        }

    except subprocess.TimeoutExpired:
        return {"error": "Generation timed out after 10 minutes"}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
