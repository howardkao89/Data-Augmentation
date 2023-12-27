import argparse
from pathlib import Path
import json
from tqdm import tqdm
import os
import shutil


def get_args_parser():
    parser = argparse.ArgumentParser("Visualization", add_help=False)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--input_dir", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--prompt", default="", type=str, required=True)

    return parser


def main(args):
    num_label = {
        "fish": 0,
        "jellyfish": 0,
        "penguin": 0,
        "puffin": 0,
        "shark": 0,
        "starfish": 0,
        "stingray": 0,
    }

    with open(args.input_file, "r") as input_file:
        input_list = json.load(input_file)

    for image_dict in tqdm(input_list):
        if num_label[image_dict["label"]] < 5:
            num_label[image_dict["label"]] = num_label[image_dict["label"]] + 1
        else:
            continue

        Path(
            os.path.join(
                args.output_dir,
                args.prompt,
                image_dict["label"],
            )
        ).mkdir(parents=True, exist_ok=True)

        shutil.copyfile(
            os.path.join(
                args.input_dir,
                args.prompt,
                "gligen_image_" + image_dict["image"],
            ),
            os.path.join(
                args.output_dir,
                args.prompt,
                image_dict["label"],
                "gligen_image_" + image_dict["image"],
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualization", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
