import os
import json
import ast
from PIL import Image, ImageDraw
from PIL import ImageColor
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def parse_json(json_output):
    """??JSON??,??Markdown??"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height, save_path):
    """????????????"""
    img = im
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
        'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
        'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
    ] + additional_colors

    # ??JSON
    parsed_json = parse_json(bounding_boxes)
    
    try:
        # ????JSON
        json_output = ast.literal_eval(parsed_json)
    except Exception as e:
        # ??????,??????
        print(f"JSON????: {e}, ??????...")
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # ?????
    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]
        abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)

    img.save(save_path)
    print(f"Saved annotated image to {save_path}")

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
    """??????"""
    image = Image.open(img_url)
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "image": img_url
                }
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0], input_height, input_width

def process_folder(folder_path, prompt, txt_folder, image_folder):
    """???????????,??JSON?txt???????"""
    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            
            # ????
            response, input_height, input_width = inference(image_path, prompt)
            
            # ??JSON??
            parsed_json = parse_json(response)
            
            # ??JSON?txt??
            txt_path = os.path.join(txt_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(txt_path, 'w') as txt_file:
                txt_file.write(parsed_json)
            print(f"Saved parsed JSON to {txt_path}")
            
            # ?????????
            image = Image.open(image_path)
            image.thumbnail([640, 640], Image.Resampling.LANCZOS)
            save_path = os.path.join(image_folder, f"annotated_{filename}")
            plot_bounding_boxes(image, response, input_width, input_height, save_path)

# ????
folder_path = "CVC-300"
txt_folder = "CVC-300_TXT"
image_folder = "CVC-300_Annotated_Images"
prompt = "Outline the position of each colorectal polyp in the endoscopic image and output all the coordinates in JSON format."
process_folder(folder_path, prompt, txt_folder, image_folder)