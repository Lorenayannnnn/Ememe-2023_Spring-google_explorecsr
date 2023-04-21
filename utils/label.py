import os
import json
import random

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