import os
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

def get_device(status=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def train_epoch(
        model,
        loader,
        optimizer,
        device,
        index_2_emotion_class,
        training=True,
):
    if training:
        model.train()
    else:
        model.eval()
    epoch_loss = 0.0
    epoch_contrastive_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []
    # put model inputs to device
    model = model.to(device)

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for loaded_inputs in tqdm(loader):
        emoroberta_inputs = loaded_inputs["emoroberta_input"]
        vilt_inputs = loaded_inputs["vilt_input"]
        labels = loaded_inputs["labels"].to(device)

        # emoroberta_inputs, vilt_inputs, labels = emoroberta_inputs.to(device), vilt_inputs.to(device), labels.to(
        #     device).long()

        # calculate the loss and train accuracy and perform backprop
        outputs = model(emoroberta_inputs, vilt_inputs, labels, device)
        loss = outputs.loss
        pred_logits = outputs.logits
        contrastive_loss = outputs.contrastive_loss

        # logging
        epoch_loss += loss.item()
        epoch_contrastive_loss += contrastive_loss.item()

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # compute metrics
        preds = pred_logits.detach().argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)
    epoch_contrastive_loss /= len(loader)

    # def report_classification_accuracy(pred_labels, true_labels):
    #     from sklearn.metrics import confusion_matrix
    #     cm = confusion_matrix(pred_labels, true_labels)
    #
    #     print(cm)
    #
    #     import numpy as np
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #
    #     acc_results = cm.diagonal()
    #     emotion_class_2_acc = {}
    #     for idx, class_name in enumerate(index_2_emotion_class):
    #         emotion_class_2_acc[class_name] = acc_results[idx]
    #         # print(class_name, acc_results[idx])
    #     return emotion_class_2_acc

    # emotion_class_2_acc = report_classification_accuracy(pred_labels=pred_labels, true_labels=target_labels)
    report = classification_report(y_pred=pred_labels, y_true=target_labels)

    return epoch_loss, acc, report, epoch_contrastive_loss


def validate(model, loader, optimizer, device, index_2_emotion_class) -> (float, float, dict):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc, val_report, val_contrastive_loss = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            training=False,
            index_2_emotion_class=index_2_emotion_class
        )

    return val_loss, val_acc, val_report, val_contrastive_loss


def train(num_epochs, model, loaders, optimizer, device, index_2_emotion_class, output_dir):
    best_val_acc = 0
    for epoch in range(num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc, train_report, train_contrastive_loss = train_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            index_2_emotion_class=index_2_emotion_class
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        val_loss, val_acc, val_report, val_contrastive_loss = validate(
            model=model,
            loader=loaders["val"],
            optimizer=optimizer,
            device=device,
            index_2_emotion_class=index_2_emotion_class
        )
        print(f"val loss : {val_loss} | val contrastive loss: {val_contrastive_loss} | val acc: {val_acc}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            ckpt_model_file = os.path.join(output_dir, "model.ckpt")
            performance_file = os.path.join(output_dir, "results.txt")
            print("saving model to ", ckpt_model_file)
            torch.save(model, ckpt_model_file)
            with open(performance_file, 'a') as writer:
                writer.write(f"Epoch: {epoch} | Train acc: {train_acc} | Dev acccuracy: {best_val_acc}\n")
                writer.write(f"Epoch: {epoch} | Train loss: {train_loss} | Dev acccuracy: {val_loss}\n")
                writer.write(val_report)
                writer.write('\n')
                # for (emotion_class, acc) in val_emotion_class_2_acc.items():
                #     writer.write(f"{emotion_class}: {acc}\n")


def setup_optimizer(lr, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    print("check model on device: ", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    return optimizer
