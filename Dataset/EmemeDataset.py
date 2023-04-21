import json
import os
from torch.utils import data
from transformers import ViltProcessor, AutoTokenizer

import torch
import numpy as np
from PIL import Image
import pickle as pkl
import requests


# Reference: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/ViLT/Fine_tuning_ViLT_for_VQA.ipynb#scrollTo=Dl2UsPrTHbtu
class EmemeDataset(data.Dataset):
    """
    EmemeDataset
    """

    def __init__(self, data_dir, json_file_path_list, processor, split, emotion_list, out_file_name: str):
        self.split = split
        self.data_dir = data_dir
        self.json_file_path_list = json_file_path_list
        self.processor = processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            "arpanghoshal/EmoRoBERTa",
            cache_dir="ememe_emoroberta_cache/"
        )
        self.cached_data_file = '{}_{}.pkl'.format(out_file_name, "dev" if split == "test" else "train")
        self.emotion_list = emotion_list
        self.emotion2label = {item: index for index, item in enumerate(emotion_list)}
        self.emotion_times = {item: 0 for index, item in enumerate(emotion_list)}
        self.num_labels = len(emotion_list)
        print(f"cached_data_file in {self.cached_data_file}")
        print("num_labels: ", self.num_labels)

        if os.path.exists(self.cached_data_file):
            # Load cached raw_json_data
            self.data = pkl.load(open(self.cached_data_file, 'rb'))
        else:
            self.data = []
            for json_file in json_file_path_list:
                json_file_path = os.path.join(data_dir, json_file)
                # print("json_file_path: ", json_file_path)
                with open(json_file_path) as file:
                    if split == 'train':
                        need = 500
                    else:
                        need = 50
                    j = json.load(file)
                    for entry in j:
                        image_url_list = entry["photos"]
                        # print("image_url_list: ", image_url_list)
                        image_url = image_url_list[0]
                        # for image_url in image_url_list:
                        try:
                            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                            fixed_size = (384, 384)
                            image_resized = image.resize(fixed_size, Image.ANTIALIAS)
                            image = image_resized
                        except Exception:
                            emotion = entry["emotion"]
                            print(f"Error when reading {image_url} with emotion {emotion}")
                            continue
                        example = {
                            'text': entry["tweet"],
                            'image_url': image_url,
                            'emotion': entry["emotion"],
                            'label': self.emotion2label[entry["emotion"]],
                            'image': image_resized
                        }
                        self.emotion_times[entry["emotion"]] += 1
                        self.data.append(example)
                        need -= 1
                        if (need%10 == 0):
                            print(need)
                        if (need <= 0):
                            break

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        self.n_examples = len(self.data)
        print("Loaded tweets {} dataset, with {} examples".format(self.split, len(self.data)))
        json.dump(self.emotion_times, open('emotion_times.json', 'w'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example['text']
        image_url = example['image_url']
        emotion = example['emotion']
        label = example['label']
        image = example['image']

        vilt_encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        encoding = {}
        encoding['vilt_input'] = {}
        encoding['emoroberta_input'] = {}

        # remove batch dimension
        for k, v in vilt_encoding.items():
            encoding['vilt_input'][k] = v.squeeze()
        # add labels
        encoding['labels'] = torch.tensor(np.array(label))
        emoroberta_input = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        for k, v in emoroberta_input.items():
            encoding['emoroberta_input'][k] = v.squeeze()

        return encoding
        # except Exception:
        #     print(f"Error when reading {image_url} with emotion {emotion}")

def batch_collate(batch):
    vilt_input_ids = torch.stack([x['vilt_input']['input_ids'] for x in batch])
    vilt_token_type_ids = torch.stack([x['vilt_input']['token_type_ids'] for x in batch])
    vilt_attention_mask = torch.stack([x['vilt_input']['attention_mask'] for x in batch])
    vilt_pixel_values = torch.stack([x['vilt_input']['pixel_values'] for x in batch])
    vilt_pixel_mask = torch.stack([x['vilt_input']['pixel_mask'] for x in batch])
    emoroberta_input_ids = torch.stack([x['emoroberta_input']['input_ids'] for x in batch])
    emoroberta_attention_mask = torch.stack([x['emoroberta_input']['attention_mask'] for x in batch])
    labels = torch.stack([x['labels'] for x in batch])

    return {
        'vilt_input':{
            'input_ids': vilt_input_ids,
            'token_type_ids': vilt_token_type_ids,
            'attention_mask': vilt_attention_mask,
            'pixel_values': vilt_pixel_values,
            'pixel_mask': vilt_pixel_mask
        },
        'emoroberta_input':{
            'input_ids': emoroberta_input_ids,
            'attention_mask': emoroberta_attention_mask
        },
        'labels': labels
    }


def build_ememe_dataloader(batch_size: int,
                           data_dir: str,
                           split: str,
                           emotion_classes,
                           **kwargs) -> torch.utils.data.DataLoader:
    shuffle = split == 'train'
    dataloader = torch.utils.data.DataLoader(build_ememe_dataset(batch_size, data_dir, split, emotion_classes),
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             collate_fn=lambda x: batch_collate(x)
                                             )
    return dataloader


def build_ememe_dataset(batch_size: int, data_dir: str, split: str, emotion_classes: list, out_file_name: str,
                        **kwargs) -> EmemeDataset:
    print("Creating tweets {} dataloader with batch size of {}".format(split, batch_size))
    json_file_path_list = [file for file in os.listdir(data_dir) if
                           os.path.isfile(os.path.join(data_dir, file)) and file.endswith('.json') and split in file]
    print("json_file_path_list: ", json_file_path_list)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    return EmemeDataset(data_dir, json_file_path_list, processor, split, emotion_classes, out_file_name)


if __name__ == "__main__":
    emotion_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    # train_dataloader = build_ememe_dataloader(64, 'data', 'train', emotion_list)
    # pkl.dump(train_dataloader, open('cached_train_dataset.pkl', 'wb'))
    # test_dataloader = build_ememe_dataloader(64, 'data', 'test', emotion_list)
    # pkl.dump(test_dataloader, open('cached_test_dataset.pkl', 'wb'))
    cached_data_filename = 'cached_data'
    # batch_size: int, data_dir: str, split: str, emotion_classes: list, out_file_name
    train_dataset = build_ememe_dataset(
        batch_size=64,
        data_dir='data',
        split='train',
        emotion_classes=emotion_list,
        out_file_name=cached_data_filename)
    pkl.dump(train_dataset, open('cached_data_train.pkl', 'wb'))
    dev_dataset = build_ememe_dataset(
        batch_size=64,
        data_dir='data',
        split='test',
        emotion_classes=emotion_list,
        out_file_name=cached_data_filename
    )
    pkl.dump(dev_dataset, open('cached_data_dev.pkl', 'wb'))
