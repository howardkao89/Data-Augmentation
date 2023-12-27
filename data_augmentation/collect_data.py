import argparse
from pathlib import Path
import json
from tqdm import tqdm
import os
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser("Collect Data", add_help=False)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--original_image_dir", default="", type=str, required=True)
    parser.add_argument("--generated_image_dir", default="", type=str, required=True)
    parser.add_argument("--prompt", default="", type=str, required=True)
    parser.add_argument("--text", default=False, action="store_true")
    parser.add_argument("--image", default=False, action="store_true")

    return parser


def main(args):
    with open(args.input_file, "r") as input_file:
        input_list = json.load(input_file)

    num_label = {
        "fish": 0,
        "jellyfish": 0,
        "penguin": 0,
        "puffin": 0,
        "shark": 0,
        "starfish": 0,
        "stingray": 0,
    }

    for image_dict in tqdm(input_list):
        if num_label[image_dict["label"]] < 20:
            num_label[image_dict["label"]] = num_label[image_dict["label"]] + 1
        else:
            continue

        if args.text:
            Path(os.path.join(args.output_dir, "text", args.prompt, "original")).mkdir(
                parents=True, exist_ok=True
            )
            Image.open(
                os.path.join(args.original_image_dir, image_dict["image"])
            ).resize((512, 512)).save(
                os.path.join(
                    args.output_dir,
                    "text",
                    args.prompt,
                    "original",
                    image_dict["image"],
                )
            )
            Path(os.path.join(args.output_dir, "text", args.prompt, "generated")).mkdir(
                parents=True, exist_ok=True
            )
            Image.open(
                os.path.join(
                    args.generated_image_dir,
                    "text",
                    args.prompt,
                    "gligen_text_" + image_dict["image"],
                )
            ).resize((512, 512)).save(
                os.path.join(
                    args.output_dir,
                    "text",
                    args.prompt,
                    "generated",
                    "gligen_text_" + image_dict["image"],
                )
            )
        elif args.image:
            Path(os.path.join(args.output_dir, "image", args.prompt, "original")).mkdir(
                parents=True, exist_ok=True
            )
            Image.open(
                os.path.join(args.original_image_dir, image_dict["image"])
            ).resize((512, 512)).save(
                os.path.join(
                    args.output_dir,
                    "image",
                    args.prompt,
                    "original",
                    image_dict["image"],
                )
            )
            Path(
                os.path.join(args.output_dir, "image", args.prompt, "generated")
            ).mkdir(parents=True, exist_ok=True)
            Image.open(
                os.path.join(
                    args.generated_image_dir,
                    "image",
                    args.prompt,
                    "gligen_image_" + image_dict["image"],
                )
            ).resize((512, 512)).save(
                os.path.join(
                    args.output_dir,
                    "image",
                    args.prompt,
                    "generated",
                    "gligen_image_" + image_dict["image"],
                )
            )

    print(f"# of image for each label: {num_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect Data", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
