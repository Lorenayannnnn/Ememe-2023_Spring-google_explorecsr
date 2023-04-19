# Ememe-2023_Spring-google_explorecsr
Ememe Project for google explore csr 2023

1. cd to project directory
  ```
  cd Ememe-2023_Spring-google_explorecsr
  ```
2. Install requirements:
  ```
  pip install -r requirements.txt
  ```
3. Prepare dataset:
  ```
  python3 Dataset/EmemeDataset.py
  ```
4. Download pretrained checkpoint of EmoROBERTa from [here](https://huggingface.co/tae898/emoberta-base) and place it under the project root directory. 
5. Go to [run_train.sh](run_train.sh) script and change the value after ```emoroberta_model_ckpt``` to the file path/name you downloaded for step 4. 
6. Run train:
  ```
  bash run_train.sh
  ```