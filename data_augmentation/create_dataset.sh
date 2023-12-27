python create_dataset.py \
    --original_input_file hw1_dataset/annotations/train.json \
    --generated_input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --input_dir hw1_dataset \
    --output_dir image_dataset \
    --generated_image_dir result/gligen \
    --prompt prompt_w_suffix \
    --image
