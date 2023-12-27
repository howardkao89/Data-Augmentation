# CSIE5428 Computer Vision Practice with Deep Learning Homework 3
* Name: 高榮浩
* ID: R12922127

## Environments

* Ubuntu 20.04
* GeForce RTX™ 2080 Ti 11G
* Python 3.8
* CUDA 11.8

```sh
   pip install -r requirements.txt
```

## Run
### BLIP-2
```sh
bash bilp2.sh
```

```sh
python blip2.py \
    --model_name Salesforce/blip2-opt-6.7b-coco \
    --input_file hw1_dataset/annotations/train.json \
    --output_file result/blip2/blip2-opt-6.7b-coco.json \
    --image_dir hw1_dataset/train

```

### GLIGEN
#### Text Grounding
```sh
bash gligen_text.sh
```

```sh
python gligen_text.py \
    --model_name masterful/gligen-1-4-generation-text-box \
    --input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --output_dir result/gligen/text/prompt_w_label \
    --prompt prompt_w_label

```

#### Image Grounding
```sh
bash gligen_image.sh
```

```sh
python gligen_image.py \
    --model_name anhnct/Gligen_Text_Image \
    --input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --output_dir result/gligen/image/prompt_w_suffix \
    --image_dir hw1_dataset/train \
    --prompt prompt_w_suffix

```

### Collect Data (FID)
```sh
bash collect_data.sh
```

```sh
python collect_data.py \
    --input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --output_dir fid \
    --original_image_dir hw1_dataset/train \
    --generated_image_dir result/gligen \
    --prompt prompt_w_suffix \
    --image

```

### Create Dataset (Data Augmentation)
```sh
bash create_dataset.sh
```

```sh
python create_dataset.py \
    --original_input_file hw1_dataset/annotations/train.json \
    --generated_input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --input_dir hw1_dataset \
    --output_dir image_dataset \
    --generated_image_dir result/gligen \
    --prompt prompt_w_suffix \
    --image

```

### Visualization
```sh
bash visualization.sh
```

```sh
python visualization.py \
    --input_file result/blip2/blip2-opt-6.7b-coco_revised.json \
    --input_dir result/gligen/image \
    --output_dir visualization \
    --prompt prompt_w_suffix

```
