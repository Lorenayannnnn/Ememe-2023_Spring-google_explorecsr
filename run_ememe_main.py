
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate as evaluate
import numpy as np
import torch
import transformers
from datasets import load_dataset
import pickle as pkl

from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    TrainingArguments, set_seed, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    default_data_collator, DataCollatorWithPadding, Trainer
)
import datasets
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction

from models.EmoRobertaForEmeme import EmoRobertaForEmeme
from models.ViLTForEmeme import ViLTForMemeSentimentClassification
from models.model import EmemeModel

# Do not remove
from Dataset.EmemeDataset import EmemeDataset
from train_utils import setup_optimizer, train, validate

dataset_idx_to_label = {
    "ememe": {
        "0": "anger",
        "1": "disgust",
        "2": "fear",
        "3": "joy",
        "4": "sadness",
        "5": "surprise"
    },
    "goemotion": {
        "0": "admiration",
        "1": "amusement",
        "2": "anger",
        "3": "annoyance",
        "4": "approval",
        "5": "caring",
        "6": "confusion",
        "7": "curiosity",
        "8": "desire",
        "9": "disappointment",
        "10": "disapproval",
        "11": "disgust",
        "12": "embarrassment",
        "13": "excitement",
        "14": "fear",
        "15": "gratitude",
        "16": "grief",
        "17": "joy",
        "18": "love",
        "19": "nervousness",
        "20": "optimism",
        "21": "pride",
        "22": "realization",
        "23": "relief",
        "24": "remorse",
        "25": "sadness",
        "26": "surprise",
        "27": "neutral"
    },
    "memotion": {
        "1": "happiness",
        "2": "love",
        "3": "anger",
        "4": "sorrow",
        "5": "fear",
        "6": "hate",
        "7": "surprise"
    }
}

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    emoroberta_model_ckpt: str = field(
        metadata={"help": "Filename to emoroberta model checkpoint"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    ),
    emoroberta_model_name_or_path: str = field(
        default="arpanghoshal/EmoRoBERTa",
        metadata={"help": "for EmoRoBERTa: Path to pretrained model or model identifier from huggingface.co/models"}
    )
    vilt_model_name_or_path: str = field(
        default="dandelin/vilt-b32-mlm",
        metadata={"help": "for ViLT: Path to pretrained model or model identifier from huggingface.co/models"}
    )
    loss_c: float = field(
        default=0.1,
        metadata={"help": "percentage of classification loss that will be added in addition to contrastive loss"}
    )
    contrastive_logit_scale: float = field(
        default=2.6592,
        metadata={"help": "logit scale for contrastive loss"}
    )
    projection_dim: int = field(
        default=512,
        metadata={"help": "output dimension of projection layer of the ememe model"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what raw_json_data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training raw_json_data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation raw_json_data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test raw_json_data."})
    cached_dataset: Optional[str] = field(
        default=None, metadata={"help": "locally cached ememe dataset pickle file"}
    )
    train_w_huggingface_trainer: Optional[bool] = field(default=True, metadata={"help": "whether use Huggingface trainer or not"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets
    if data_args.dataset_name is not None:
        if data_args.dataset_name == "ememe":
            # Load from preprocessed local data pkl file
            ememe_train_datafile = "{}_train.pkl".format(data_args.cached_dataset)
            ememe_dev_datafile = "{}_dev.pkl".format(data_args.cached_dataset)
            raw_datasets = {
                "train": pkl.load(open(ememe_train_datafile, "rb")),
                "validation": pkl.load(open(ememe_dev_datafile, "rb"))
            }
        else:
            raise ValueError("Dataset not supported yet")
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                        test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.dataset_name is not None:
        num_labels = len(list(dataset_idx_to_label[data_args.dataset_name].keys())) if data_args.dataset_name in dataset_idx_to_label.keys() else 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    '''
    Load pretrained model and tokenizer
    '''
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if data_args.dataset_name != "ememe":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if data_args.dataset_name == "ememe":
        emoroberta_config = AutoConfig.from_pretrained(
            model_args.emoroberta_model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        vilt_config = AutoConfig.from_pretrained(
            model_args.vilt_model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if data_args.dataset_name is not None:
        label_list = list(dataset_idx_to_label[data_args.dataset_name].values())
        label_to_id = {v: i for i, v in enumerate(label_list)}
        if data_args.dataset_name == "ememe":
            emoroberta_config.label2id = label_to_id
            emoroberta_config.id2label = dataset_idx_to_label[data_args.dataset_name]
            vilt_config.label2id = label_to_id
            vilt_config.id2label = dataset_idx_to_label[data_args.dataset_name]
        else:
            config.label2id = label_to_id
            config.id2label = dataset_idx_to_label[data_args.dataset_name]

    if model_args.model_name_or_path:
        model_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
            "ignore_mismatched_sizes": model_args.ignore_mismatched_sizes
        }
        if data_args.dataset_name == "goemotion":
            # EmoRoBERTa
            model = EmoRobertaForEmeme(
                config=config,
                model_name_or_path=model_args.emoroberta_model_ckpt if model_args.emoroberta_model_ckpt else model_args.emoroberta_model_name_or_path,
                model_kwargs=model_kwargs
            )
        elif data_args.dataset_name == "memotion":
            # processor = ViltProcessor.from_pretrained(
            #     model_args.model_name_or_path,
            #     config=config,
            #     **model_kwargs
            # )
            model = ViLTForMemeSentimentClassification(
                model_name_or_path=model_args.model_name_or_path,
                config=config,
                **model_kwargs
            )
        elif data_args.dataset_name == "ememe":
            text_model = EmoRobertaForEmeme(
                config=emoroberta_config,
                model_name_or_path=model_args.emoroberta_model_name_or_path,
                model_kwargs=model_kwargs
            )
            # processor = ViltProcessor.from_pretrained(
            #     model_args.vilt_model_name_or_path,
            #     config=vilt_config,
            #     **model_kwargs
            # )
            meme_model = ViLTForMemeSentimentClassification(
                model_name_or_path=model_args.vilt_model_name_or_path,
                config=vilt_config,
                model_kwargs=model_kwargs
            )
            model = EmemeModel(
                text_model=text_model,
                meme_model=meme_model,
                loss_c=model_args.loss_c,
                contrastive_logit_scale=model_args.contrastive_logit_scale,
                projection_dim=model_args.projection_dim
            )
        else:
            # model = AutoModel.from_pretrained(
            #     model_args.model_name_or_path,
            #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
            #     config=config,
            #     **model_kwargs
            # )
            raise ValueError("Model for the input task is not supported")
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSequenceClassification.from_config(config)

    model = model.to(training_args.device)

    if data_args.dataset_name != "ememe":
        '''
        Preprocess Dataset
        '''
        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        def preprocess_function(examples):
            # TODO: update example access based on dataset loading
            if data_args.dataset_name == "goemotion":
                args = (
                    examples["text"]
                )
                result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True, return_tensors="pt")
            elif data_args.dataset_name == "ememe":
                args = (
                    examples["image"],
                    examples["text"]
                )
                result = processor(*args, padding=padding, max_length=max_seq_length, truncation=True, return_tensors="pt")

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        if data_args.dataset_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training
    if data_args.train_w_huggingface_trainer:
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            output_train_file = os.path.join(training_args.output_dir, f"train_results_{data_args.dataset_name}.txt")
            if trainer.is_world_process_zero():
                with open(output_train_file, "w") as writer:
                    logger.info("***** Train results *****")
                    for key, value in sorted(metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{data_args.dataset_name}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
    else:
        # Set up dataloader
        loaders = {
            "train": DataLoader(train_dataset, shuffle=True, batch_size=training_args.per_device_train_batch_size),
            "val": DataLoader(eval_dataset, shuffle=True, batch_size=training_args.per_device_eval_batch_size),
        }

        # Set up optimizer
        optimizer = setup_optimizer(training_args.learning_rate, model)

        if training_args.do_train:
            train(
                num_epochs=training_args.num_train_epochs,
                model=model,
                loaders=loaders,
                optimizer=optimizer,
                device=training_args.device,
                index_2_emotion_class=dataset_idx_to_label[data_args.dataset_name],
                output_dir=training_args.output_dir
            )
        elif training_args.do_eval:
            # Load pretrained model
            model = torch.load(os.path.join(training_args.output_dir, "model.ckpt"))
            val_loss, val_acc = validate(
                model=model,
                loader=loaders["val"],
                optimizer=optimizer,
                device=training_args.device,
                index_2_emotion_class=dataset_idx_to_label[data_args.dataset_name]
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")


if __name__ == "__main__":
    main()
