"""End-to-end reproduction of MagMax (Marczak et al., ECCV 2024) on the
8-task CLIP image-classification benchmark.

This script faithfully follows the *sequential* fine-tuning protocol from
``tmp/magmax/finetune_8datasets.py`` (the paper's official code) and then
applies the FusionBench MagMax merger on the resulting per-task snapshots.

Pipeline
--------
1. Load CLIP (HF transformers) and freeze the text encoder + visual
   projection + logit scale.
2. Build a frozen zero-shot classification head for each of the 8 tasks
   using ``get_classnames_and_templates``.
3. Iterate through the tasks in the paper's order
   (Cars -> MNIST -> EuroSAT -> SVHN -> RESISC45 -> SUN397 -> DTD -> GTSRB).
   For each task, continue fine-tuning the vision tower from the previous
   task's weights for ``num_steps_per_task`` iterations with AdamW + cosine
   LR and gradient clipping.
4. After each task, save the CLIPVisionModel to
   ``<output_dir>/<task_idx>_<task_name>/`` in HuggingFace format.
5. After all tasks, run :class:`fusion_bench.method.MagMaxAlgorithm` over the
   8 snapshots and evaluate the merged model on the TA8 test sets.

Usage
-----
::

    .venv/bin/python examples/magmax/sequential_ft_and_merge.py \\
        --base-model openai/clip-vit-base-patch32 \\
        --output-dir outputs/magmax/seq_ft_b32 \\
        --num-steps-per-task 2000 \\
        --scaling-factor 0.5

To skip training (e.g. resume after a crash) and just rerun the MagMax merge
on existing snapshots, pass ``--skip-training``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

from fusion_bench.dataset.image_dataset import ImageClassificationDataset
from fusion_bench.method.magmax import magmax_merge
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.tasks.clip_classification import get_classnames_and_templates

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("magmax.seq_ft")


# Task order from the official ``finetune_8datasets.py``. Names here match
# FusionBench's dataset registry keys so that ``get_classnames_and_templates``
# and the HF dataset loader keep working unchanged.
TASKS: List[Tuple[str, dict]] = [
    ("stanford-cars", {"path": "tanganke/stanford_cars"}),
    ("mnist", {"path": "mnist"}),
    ("eurosat", {"path": "tanganke/eurosat"}),
    ("svhn", {"path": "svhn", "name": "cropped_digits"}),
    ("resisc45", {"path": "tanganke/resisc45"}),
    ("sun397", {"path": "tanganke/sun397"}),
    ("dtd", {"path": "tanganke/dtd"}),
    ("gtsrb", {"path": "tanganke/gtsrb"}),
]


def _load_split(spec: dict, split: str):
    """Load a HF dataset following the same conventions as the existing
    FusionBench dataset configs."""
    if "name" in spec:
        return load_dataset(spec["path"], spec["name"], split=split)
    return load_dataset(spec["path"], split=split)


def _build_classifier_heads(
    base_model: str,
    device: torch.device,
) -> Tuple[CLIPProcessor, Dict[str, torch.Tensor], torch.Tensor, float]:
    """Pre-compute frozen zero-shot text embeddings for each task.

    Returns the CLIP processor, a dict of zero-shot weights per task, the
    visual projection (a Linear layer mapping vision-tower features into the
    shared CLIP embedding space) and the scalar ``logit_scale.exp()``.
    """
    log.info("Building frozen zero-shot classifier heads from %s", base_model)
    full_clip = CLIPModel.from_pretrained(base_model).to(device).eval()
    processor = CLIPProcessor.from_pretrained(base_model)
    classifier = HFCLIPClassifier(full_clip, processor)

    zeroshot_weights: Dict[str, torch.Tensor] = {}
    for task_name, _ in TASKS:
        classnames, templates = get_classnames_and_templates(task_name)
        classifier.set_classification_task(classnames, templates)
        zeroshot_weights[task_name] = classifier.zeroshot_weights.detach().clone()

    visual_projection = deepcopy(full_clip.visual_projection).requires_grad_(False)
    logit_scale = float(full_clip.logit_scale.detach().exp().item())

    # Free the text tower; we only need the vision tower from here on.
    del full_clip, classifier
    torch.cuda.empty_cache()
    return processor, zeroshot_weights, visual_projection, logit_scale


def _vision_logits(
    vision_model: CLIPVisionModel,
    visual_projection: nn.Linear,
    images: torch.Tensor,
    text_weights: torch.Tensor,
    logit_scale: float,
) -> torch.Tensor:
    """Compute CLIP zero-shot logits for a single batch."""
    pooled = vision_model(pixel_values=images).pooler_output
    image_embeds = visual_projection(pooled)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return logit_scale * image_embeds @ text_weights.t()


def _train_one_task(
    vision_model: CLIPVisionModel,
    visual_projection: nn.Linear,
    text_weights: torch.Tensor,
    logit_scale: float,
    processor: CLIPProcessor,
    train_spec: dict,
    task_name: str,
    num_steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    num_workers: int,
) -> CLIPVisionModel:
    """Fine-tune ``vision_model`` for ``num_steps`` iterations on a task."""
    train_ds = _load_split(train_spec, "train")
    dataset = ImageClassificationDataset(train_ds, processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    text_weights = text_weights.to(device)
    visual_projection.to(device)
    vision_model.to(device).train()

    optimizer = torch.optim.AdamW(
        [p for p in vision_model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(total=num_steps, desc=f"FT {task_name}", dynamic_ncols=True)
    step = 0
    while step < num_steps:
        for batch in loader:
            if step >= num_steps:
                break
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = _vision_logits(
                vision_model, visual_projection, images, text_weights, logit_scale
            )
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in vision_model.parameters() if p.requires_grad], grad_clip
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 20 == 0:
                pbar.set_postfix(loss=float(loss.item()), lr=scheduler.get_last_lr()[0])
            pbar.update(1)
    pbar.close()
    return vision_model.eval()


def _evaluate(
    vision_model: CLIPVisionModel,
    visual_projection: nn.Linear,
    zeroshot_weights: Dict[str, torch.Tensor],
    logit_scale: float,
    processor: CLIPProcessor,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, float]:
    """Evaluate the (possibly merged) vision model on all 8 test sets."""
    vision_model.to(device).eval()
    visual_projection.to(device)
    results: Dict[str, float] = {}
    with torch.no_grad():
        for task_name, spec in TASKS:
            test_ds = _load_split(spec, "test")
            loader = DataLoader(
                ImageClassificationDataset(test_ds, processor),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            text_w = zeroshot_weights[task_name].to(device)
            correct = total = 0
            for images, labels in tqdm(loader, desc=f"eval {task_name}", leave=False):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = _vision_logits(
                    vision_model, visual_projection, images, text_w, logit_scale
                )
                correct += int((logits.argmax(-1) == labels).sum().item())
                total += int(labels.numel())
            acc = correct / max(total, 1)
            log.info("eval %s: %.4f", task_name, acc)
            results[task_name] = acc
    results["average"] = sum(results.values()) / len(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", default="outputs/magmax/seq_ft_b32")
    parser.add_argument("--num-steps-per-task", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--scaling-factor", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Reuse existing per-task checkpoints under output-dir.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("Using device: %s, output dir: %s", device, args.output_dir)

    processor, zeroshot_weights, visual_projection, logit_scale = (
        _build_classifier_heads(args.base_model, device)
    )

    pretrained_dir = os.path.join(args.output_dir, "_pretrained_")
    if not os.path.isdir(pretrained_dir):
        log.info("Saving pretrained snapshot to %s", pretrained_dir)
        CLIPVisionModel.from_pretrained(args.base_model).save_pretrained(pretrained_dir)

    # Sequential FT phase
    vision_model: CLIPVisionModel = CLIPVisionModel.from_pretrained(args.base_model)
    for idx, (task_name, spec) in enumerate(TASKS):
        ckpt_dir = os.path.join(args.output_dir, f"{idx:02d}_{task_name}")
        if args.skip_training and os.path.isdir(ckpt_dir):
            log.info("[%d/%d] skip-training: reloading %s", idx + 1, len(TASKS), ckpt_dir)
            vision_model = CLIPVisionModel.from_pretrained(ckpt_dir)
            continue

        log.info("[%d/%d] sequential FT on %s", idx + 1, len(TASKS), task_name)
        vision_model = _train_one_task(
            vision_model=vision_model,
            visual_projection=visual_projection,
            text_weights=zeroshot_weights[task_name],
            logit_scale=logit_scale,
            processor=processor,
            train_spec=spec,
            task_name=task_name,
            num_steps=args.num_steps_per_task,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            device=device,
            num_workers=args.num_workers,
        )
        log.info("Saving snapshot to %s", ckpt_dir)
        vision_model.save_pretrained(ckpt_dir)

    # Merge phase
    log.info("Loading 8 sequential snapshots for MagMax merge")
    finetuned: List[CLIPVisionModel] = [
        CLIPVisionModel.from_pretrained(
            os.path.join(args.output_dir, f"{idx:02d}_{task_name}")
        )
        for idx, (task_name, _) in enumerate(TASKS)
    ]
    pretrained = CLIPVisionModel.from_pretrained(pretrained_dir)
    log.info("Running MagMax (scaling_factor=%s)", args.scaling_factor)
    merged = magmax_merge(
        pretrained, finetuned, scaling_factor=args.scaling_factor, inplace=True
    )
    merged_dir = os.path.join(args.output_dir, f"merged_alpha{args.scaling_factor}")
    merged.save_pretrained(merged_dir)
    log.info("Saved merged model to %s", merged_dir)

    # Free per-task finetuned models before eval to save GPU memory
    del finetuned
    torch.cuda.empty_cache()

    # Evaluation phase
    results = _evaluate(
        merged,
        visual_projection,
        zeroshot_weights,
        logit_scale,
        processor,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    report_path = os.path.join(
        args.output_dir, f"magmax_seqft_alpha{args.scaling_factor}.json"
    )
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Wrote report to %s", report_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
