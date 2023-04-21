import os
import json
import random

def split_dataset(data_dir, train_ratio=0.7):
    json_file_path_list = [file for file in os.listdir(data_dir) if
                           os.path.isfile(os.path.join(data_dir, file)) and file.endswith('.json')]
    train_data = []
    test_data = []
    for json_file in json_file_path_list:
        json_file_path = os.path.join(data_dir, json_file)
        file_name = json_file.replace(".json", "")
        print("json_file_path: ", json_file_path)
        with open(json_file_path) as file:
            data = json.load(file)
            random.shuffle(data)
            split_idx = int(len(data) * train_ratio)
            train_data.extend(data[:split_idx])
            test_data.extend(data[split_idx:])
        with open(f"../data/{file_name}_train_data.json", "w") as train_file:
            json.dump(train_data, train_file)

        # Save test raw_json_data to a new JSON file
        with open(f"../data/{file_name}_test_data.json", "w") as test_file:
            json.dump(test_data, test_file)

# split_dataset('../raw_json_data', train_ratio=0.7)

emotion_list = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def label(filename, emotion):
    filepath = os.path.join('data', filename)
    res_json = []
    with open(filepath, 'r') as file:
        data = json.load(file)
        for line in data:
            line['emotion'] = emotion
            res_json.append(line)
        json.dump(res_json, open('correct_'+filename, 'w'))

label('filtered_angry_test_data.json', 'anger')
label('filtered_angry_train_data.json', 'anger')
label('filtered_disgust_test_data.json', 'disgust')
label('filtered_disgust_train_data.json', 'disgust')
label('filtered_fear_test_data.json', 'fear')
label('filtered_fear_train_data.json', 'fear')
label('filtered_joy_test_data.json', 'joy')
label('filtered_joy_train_data.json', 'joy')
label('filtered_sad_test_data.json', 'sadness')
label('filtered_sad_train_data.json', 'sadness')
label('filtered_surprise_test_data.json', 'surprise')
label('filtered_surprise_train_data.json', 'surprise')

