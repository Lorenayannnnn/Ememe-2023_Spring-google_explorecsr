import json
import os
from torch.utils import data
from transformers import ViltProcessor
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
        self.cached_data_file = '{}_{}.pkl'.format(out_file_name, "dev" if split == "test" else "train")
        self.emotion_list = emotion_list
        self.emotion2label = {item: index for index, item in enumerate(emotion_list)}
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
                    j = json.load(file)
                    for entry in j:
                        image_url_list = entry["photos"]
                        # print("image_url_list: ", image_url_list)
                        for image_url in image_url_list:
                            example = {
                                'text': entry["tweet"],
                                'image_url': image_url,
                                'emotion': entry["emotion"],
                                'label': self.emotion2label[entry["emotion"]]
                            }
                            self.data.append(example)

            pkl.dump(self.data, open(self.cached_data_file, 'wb'))

        self.n_examples = len(self.data)
        print("Loaded tweets {} dataset, with {} examples".format(self.split, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = example['text']
        image_url = example['image_url']
        emotion = example['emotion']
        label = example['label']

        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
            encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
            # remove batch dimension
            for k, v in encoding.items():
                encoding[k] = v.squeeze()
            # add labels
            encoding['labels'] = torch.tensor(np.array(label))

            return encoding
        except Exception:
            print(f"Error when reading {image_url} with emotion {emotion}")


def build_ememe_dataloader(batch_size: int,
                           data_dir: str,
                           split: str,
                           emotion_classes,
                           **kwargs) -> torch.utils.data.DataLoader:
    shuffle = split == 'train'
    dataloader = torch.utils.data.DataLoader(build_ememe_dataset(batch_size, data_dir, split, emotion_classes),
                                             batch_size=batch_size,
                                             shuffle=shuffle)
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
