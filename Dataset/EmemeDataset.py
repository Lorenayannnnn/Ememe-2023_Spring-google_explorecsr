
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

    def __init__(self, data_dir, json_file_path_list, processor, split, emotion_list, **kwargs):
        self.split = split
        self.data_dir = data_dir
        self.json_file_path_list = json_file_path_list
        # what is the processor for?
        self.processor = processor
        self.cached_data_file = 'cached_twitter_data_{}.pkl'.format(split)
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
                print("json_file_path: ", json_file_path)
                with open(json_file_path) as file:
                    j = json.load(file)
                    for data in j:
                        image_url_list = data["photos"]
                        print("image_url_list: ", image_url_list)
                        for image_url in image_url_list:
                            example = {
                                'text': data["tweet"],
                                'image_url': image_url,
                                'emotion': data["emotion"],
                                'label': self.emotion2label[data["emotion"]]
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
                            emotion_list,
                            **kwargs) -> torch.utils.data.DataLoader:

    shuffle = split == 'train'

    print("Creating tweets {} dataloader with batch size of {}".format(split, batch_size))
    json_file_path_list = [file for file in os.listdir(data_dir) if
                           os.path.isfile(os.path.join(data_dir, file)) and file.endswith('.json') and split in file]
    print("json_file_path_list: ", json_file_path_list)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    dataset = EmemeDataset(data_dir, json_file_path_list, processor, split, emotion_list, **kwargs)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle)
    return dataloader

emotion_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
train_dataloader = build_ememe_dataloader(64, 'data', 'train', emotion_list)
pkl.dump(train_dataloader, open('cached_train_dataset.pkl', 'wb'))
test_dataloader = build_ememe_dataloader(64, 'data', 'test', emotion_list)
pkl.dump(test_dataloader, open('cached_test_dataset.pkl', 'wb'))

