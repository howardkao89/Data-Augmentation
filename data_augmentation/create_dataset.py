import argparse
from pathlib import Path
import json
from tqdm import tqdm
import os
import shutil


def get_args_parser():
    parser = argparse.ArgumentParser("Create Dataset", add_help=False)
    parser.add_argument("--original_input_file", default="", type=str, required=True)
    parser.add_argument("--generated_input_file", default="", type=str, required=True)
    parser.add_argument("--input_dir", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--generated_image_dir", default="", type=str, required=True)
    parser.add_argument("--prompt", default="", type=str, required=True)
    parser.add_argument("--text", default=False, action="store_true")
    parser.add_argument("--image", default=False, action="store_true")

    return parser


def main(args):
    shutil.copytree(args.input_dir, args.output_dir, dirs_exist_ok=True)

    label = [
        "creatures",
        "fish",
        "jellyfish",
        "penguin",
        "puffin",
        "shark",
        "starfish",
        "stingray",
    ]
    num_label = {
        "fish": 0,
        "jellyfish": 0,
        "penguin": 0,
        "puffin": 0,
        "shark": 0,
        "starfish": 0,
        "stingray": 0,
    }

    with open(args.original_input_file, "r") as original_input_file:
        original_input_dict = json.load(original_input_file)

    for antt in tqdm(original_input_dict["annotations"]):
        num_label[label[antt["category_id"]]] = (
            num_label[label[antt["category_id"]]] + 1
        )

    print(f"# of bbox for each label before: {num_label}")
    max_num_label = num_label[max(num_label, key=num_label.get)]
    output_dict = original_input_dict

    with open(args.generated_input_file, "r") as generated_input_file:
        generated_input_list = json.load(generated_input_file)

    for image_dict in tqdm(generated_input_list):
        if num_label[image_dict["label"]] + len(image_dict["bboxes"]) <= max_num_label:
            num_label[image_dict["label"]] = num_label[image_dict["label"]] + len(
                image_dict["bboxes"]
            )
        else:
            continue

        if args.text:
            for img in original_input_dict["images"]:
                if img["file_name"] == image_dict["image"]:
                    generated_img = img.copy()
                    generated_img["id"] = len(output_dict["images"])
                    generated_img["file_name"] = "gligen_text_" + img["file_name"]
                    output_dict["images"].append(generated_img)
                    for antt in original_input_dict["annotations"]:
                        if antt["image_id"] == img["id"]:
                            generated_antt = antt.copy()
                            generated_antt["id"] = len(output_dict["annotations"])
                            generated_antt["image_id"] = generated_img["id"]
                            output_dict["annotations"].append(generated_antt)
            shutil.copyfile(
                os.path.join(
                    args.generated_image_dir,
                    "text",
                    args.prompt,
                    "gligen_text_" + image_dict["image"],
                ),
                os.path.join(
                    args.output_dir,
                    "train",
                    "gligen_text_" + image_dict["image"],
                ),
            )
        elif args.image:
            for img in original_input_dict["images"]:
                if img["file_name"] == image_dict["image"]:
                    generated_img = img.copy()
                    generated_img["id"] = len(output_dict["images"])
                    generated_img["file_name"] = "gligen_image_" + img["file_name"]
                    output_dict["images"].append(generated_img)
                    for antt in original_input_dict["annotations"]:
                        if antt["image_id"] == img["id"]:
                            generated_antt = antt.copy()
                            generated_antt["id"] = len(output_dict["annotations"])
                            generated_antt["image_id"] = generated_img["id"]
                            output_dict["annotations"].append(generated_antt)
            shutil.copyfile(
                os.path.join(
                    args.generated_image_dir,
                    "image",
                    args.prompt,
                    "gligen_image_" + image_dict["image"],
                ),
                os.path.join(
                    args.output_dir,
                    "train",
                    "gligen_image_" + image_dict["image"],
                ),
            )

    print(f"# of bbox for each label after: {num_label}")

    with open(
        os.path.join(args.output_dir, "annotations/train.json"), "w"
    ) as output_file:
        json.dump(output_dict, output_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Create Dataset", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
