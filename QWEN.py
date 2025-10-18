import os
os.environ['CUDA_VISIBLE_DEVICES']='4'

import io
from PIL import Image
import torch
from tqdm import tqdm
from datasets import load_dataset
import json

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


def process_multimodal_dataset(model, processor, dataset, max_new_tokens=1024, device="cuda", output_file="results.json"):
    results = {}

    
    
    for i in tqdm(range(len(dataset['test']['conversation']))):
        conversation_id = dataset['test']['conversation_id'][i]
        user_msgs, assistant_msgs = [], []
        
        # Extract user and assistant messages
        for conv in dataset['train']['conversation'][i]:
            if conv[0]['role'] == 'user':
                user_msgs.append(conv[0])
            else:
                assistant_msgs.append(conv[0])
        
        # Process image
        image_data = dataset['train']['images'][i][0]['bytes']
        image = Image.open(io.BytesIO(image_data)).resize((200, 200))
        
        messages_img = []
        messages_no_img = []
        label = None
        
        for idx in range(len(assistant_msgs)):
            if idx==0:
                messages_img.append({
                    'role': 'user',
                    'content': [
                        {"type": "image", "image": image},
                        {'type': 'text', 'text': user_msgs[idx]['content']}]
                })
            else:
                messages_img.append({
                    'role': 'user',
                    'content': [{'type': 'text', 'text': user_msgs[idx]['content']}]
                })
            messages_no_img.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': user_msgs[idx]['content']}]
            })
            
            if len(assistant_msgs) - 1 > idx:
                messages_img.append({
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': assistant_msgs[idx]['content']}]
                })
                messages_no_img.append({
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': assistant_msgs[idx]['content']}]
                })
            else:
                label = assistant_msgs[idx]
        
        # Process messages with images
        text_with_img = processor.apply_chat_template(messages_img, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages_img)
        inputs = processor(
            text=[text_with_img],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_img = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Process messages without images
        text_no_img = processor.apply_chat_template(messages_no_img, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages_no_img)
        inputs = processor(
            text=[text_no_img],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text_no_img = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        results[conversation_id] = {
            "output_with_image": output_text_img,
            "output_without_image": output_text_no_img,
            "label": label
        }
    
    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    #return results, output_file



# Get dataset 
ds = load_dataset("dataset")['']

# Load model 
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
adapter_path = "zera09/qwen2-7b-fin-chat"
model.load_adapter(adapter_path)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")



process_multimodal_dataset(model, processor, ds, max_new_tokens=256,output_file='qwen_complete.json')
