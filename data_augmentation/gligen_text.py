import torch
from diffusers import StableDiffusionGLIGENPipeline
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import os


def get_args_parser():
    parser = argparse.ArgumentParser("GLIGEN Text Grounding Generation", add_help=False)
    parser.add_argument("--model_name", default="", type=str, required=True)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--prompt", default="", type=str, required=True)

    return parser


def main(args):
    # Generate an image described by the prompt and
    # insert objects described by text at the region defined by bounding boxes
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.model_name,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")

    with open(args.input_file, "r") as input_file:
        input_list = json.load(input_file)

    for image_dict in tqdm(input_list):
        prompt = image_dict[args.prompt]
        boxes = image_dict["bboxes"]
        phrases = [image_dict["label"] for _ in range(len(boxes))]
        height = image_dict["height"]
        width = image_dict["width"]

        images = pipe(
            prompt=prompt,
            gligen_phrases=phrases,
            gligen_boxes=boxes,
            gligen_scheduled_sampling_beta=1,
            output_type="pil",
            num_inference_steps=50,
            height=height,
            width=width,
        ).images

        images[0].save(
            os.path.join(args.output_dir, "gligen_text_" + image_dict["image"])
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "GLIGEN Text Grounding Generation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
