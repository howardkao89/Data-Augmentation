from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("BLIP2 Image Captioning", add_help=False)
    parser.add_argument("--model_name", default="", type=str, required=True)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_file", default="", type=str, required=True)
    parser.add_argument("--image_dir", default="", type=str, required=True)

    return parser


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    with open(args.input_file, "r") as input_file:
        input_antt = json.load(input_file)

    categories = []
    for category in tqdm(input_antt["categories"]):
        categories.append(category["name"])

    output_list = []
    for image in tqdm(input_antt["images"]):
        category_ids = []
        bboxes = []
        for antt in input_antt["annotations"]:
            if antt["image_id"] == image["id"]:
                category_ids.append(antt["category_id"])
                [x_1, y_1, w, h] = antt["bbox"]
                bbox = [
                    x_1 / image["width"],
                    y_1 / image["height"],
                    (x_1 + w) / image["width"],
                    (y_1 + h) / image["height"],
                ]
                bboxes.append(bbox)
        category_ids = list(set(category_ids))
        if len(category_ids) != 1 or len(bboxes) > 6:
            continue

        output_antt = {}
        output_antt["image"] = image["file_name"]
        output_antt["label"] = categories[category_ids[0]]
        output_antt["height"] = image["height"]
        output_antt["width"] = image["width"]
        output_antt["bboxes"] = bboxes

        image_file = Image.open(os.path.join(args.image_dir, image["file_name"]))

        inputs = processor(images=image_file, return_tensors="pt").to(
            device=device, dtype=torch.bfloat16
        )

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        output_antt["generated_text"] = generated_text
        output_antt[
            "prompt_w_label"
        ] = f"{generated_text}, {categories[category_ids[0]]}, height: {image['height']}, width: {image['width']}"
        output_antt[
            "prompt_w_suffix"
        ] = f"{generated_text}, {categories[category_ids[0]]}, height: {image['height']}, width: {image['width']}, HD quality, highly detailed"

        output_list.append(output_antt)

    with open(args.output_file, "w") as output_file:
        json.dump(output_list, output_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BLIP2 Image Captioning", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_file:
        Path(os.path.dirname(args.output_file)).mkdir(parents=True, exist_ok=True)
    main(args)
