import torch

from transformers import BertTokenizer
from PIL import Image
import argparse
import matplotlib.pyplot as plt

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import numpy as np
import math 

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to image', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
args = parser.parse_args()
image_path = args.path
version = args.v
checkpoint_path = args.checkpoint

config = Config()

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
      raise NotImplementedError('No model to chose from!')
    else:
      if not os.path.exists(checkpoint_path):
        raise NotImplementedError('Give valid checkpoint path')
      print("Found checkpoint! Loading!")
      model,_ = caption.build_model(config)
      print("Loading Checkpoint...")
      checkpoint = torch.load(checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

image = Image.open(image_path)
image = coco.val_transform(image)
image = image.unsqueeze(0)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)


@torch.no_grad()
def evaluate():
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        print("================================================")
        predictions, atten_map = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        print(predicted_id[0])
        print(f"Predict_id: {predicted_id[0]} -> {tokenizer.decode(predicted_id[0], skip_special_tokens=False)}")
        if predicted_id[0] == 102:
            # Reach EOF
            # print(atten_map.shape)
            # print(atten_map)
            # print(caption)
            n_word = 0
            for j, word in enumerate(caption[0]):
                if word == 0:
                    n_word = j+1
                    break


            # idx = 0
            # while caption[0][idx] != 0:
            # ((ax1, ax2), (ax3, ax4))
            N_ROW = 4
            fig, ax_list = plt.subplots(nrows=math.ceil(n_word/N_ROW), ncols=N_ROW, figsize=(10, 10))
            print(ax_list)
            for j in range(n_word):
                single_atten_map = atten_map[0][j]
                # print (single_atten_map)
                
                # Denormalize
                single_atten_map = single_atten_map*(255.0 / torch.max(single_atten_map))
                single_atten_map = single_atten_map.reshape((19, 19))
                
                atten_img = Image.fromarray(np.array(single_atten_map, dtype=np.uint8))
                
                token = tokenizer.decode(caption[0][j].tolist(), skip_special_tokens=False).replace(" ", "")
                # atten_img.save(f"{idx}_{token}.jpg")
                # plt.imshow(image)

                img_ori = Image.open(image_path).convert('RGB')
                # print(img_ori.size)
                
                ax = ax_list[math.floor(j/N_ROW)][math.floor(j%N_ROW)]
                ax.imshow(img_ori, alpha=1)
                plt.axis('off')
                atten_img = atten_img.resize(img_ori.size)
                ax.imshow(atten_img, alpha=0.5, interpolation='nearest', cmap="jet")
                
                # plt.subplots(nrows=1, ncols=1)
                # plt.imshow(img_ori, alpha=1)
                # plt.axis('off')
                # atten_img = atten_img.resize(img_ori.size)
                # plt.imshow(atten_img, alpha=0.5, interpolation='nearest', cmap="jet")
                # plt.savefig(f"{idx}_{token}.png")
                # idx += 1
            plt.savefig("test.png")
            
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
        # print(caption)

    return caption


output = evaluate()
result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
#result = tokenizer.decode(output[0], skip_special_tokens=True)
print(result.capitalize())