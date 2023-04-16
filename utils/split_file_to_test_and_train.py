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

split_dataset('../raw_json_data', train_ratio=0.7)