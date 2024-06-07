import json
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import tqdm
from torch import Tensor

from . import utils
from .datasets.common import get_dataloader, maybe_dictionarize
from .datasets.registry import get_dataset
from .heads import ClassificationHead, get_classification_head
from .modeling import ImageClassifier, ImageEncoder

log = logging.getLogger(__name__)


def eval_single_dataset(
    image_encoder: ImageEncoder, dataset_name: str, args, dataloader=None
):
    device = args.device

    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model = model.to(device)
    model.eval()

    if dataloader is None:
        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            num_workers=args.num_workers,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=None
        )

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data: Dict[str, Tensor] = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits: Tensor = model(x)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

            if hasattr(args, "fast_dev_run") and args.fast_dev_run:
                break

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%")
    return metrics


def eval_single_dataset_head(
    image_encoder: ImageEncoder,
    head: ClassificationHead,
    dataset_name: str,
    args,
    dataloader: torch.utils.data.DataLoader = None,
):
    """
    Evaluates the performance of a given image encoder and classification head on a single dataset.

    Args:
        image_encoder (ImageEncoder): The image encoder to use for feature extraction.
        head (ClassificationHead): The classification head to use for prediction.
        dataset_name (str): The name of the dataset to evaluate on.
        args: The command-line arguments passed to the script.

    Returns:
        A dictionary containing the evaluation metrics, with the following keys:
        - "top1": The top-1 accuracy of the model on the given dataset.
    """
    if dataloader is None:
        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=None
        )

    model = ImageClassifier(image_encoder, head)
    model.eval()
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")
    return metrics


def eval_single_dataset_preprocess_head(
    image_encoder: ImageEncoder,
    head: ClassificationHead,
    dataset_name: str,
    args,
    dataloader: torch.utils.data.DataLoader = None,
):
    model = ImageClassifier(image_encoder, head)
    model = model.to(args.device, non_blocking=True)
    model.eval()

    if dataloader is None:
        dataset = get_dataset(
            dataset_name,
            model.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        dataloader = get_dataloader(
            dataset, is_train=False, args=args, image_encoder=None
        )
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for i, data in enumerate(pbar := tqdm.tqdm(dataloader, leave=False)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits: Tensor = model(x)
            pred = logits.argmax(dim=1, keepdim=True)

            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
            pbar.set_description(f"Top-1 Acc: {100 * correct / n:.2f}%")

            if args.fast_dev_run:
                log.info("fast_dev_run")
                break  # early stop if fast_dev_run, for debugging

        top1 = correct / n

    metrics = {"top1": top1}
    print(f"Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%")

    return metrics


def evaluate(image_encoder, args):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print("Evaluating on", dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if "top1" in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")
        for key, val in results.items():
            if "worst" in key or "f1" in key.lower() or "pm0" in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            info[dataset_name + ":" + key] = val

    if args.results_db is not None:
        dirname = os.path.dirname(args.results_db)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.results_db, "a+") as f:
            f.write(json.dumps(info) + "\n")
        print(f"Results saved to {args.results_db}.")
    else:
        print("Results not saved (to do so, use --results_db to specify a path).")

    return info
