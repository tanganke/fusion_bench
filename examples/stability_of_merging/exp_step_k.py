# 验证 steps K_i
#
# 计算指标：在挑选出来的任务的联合测试数据集上测出来loss 与 acc这两个指标。这里的联合是指比如挑选出来mnist和cifar10，你们测试联合数据集就是cifar10和mnist的测试数据集的并集。
# 具体做法：直接将不同的steps按照排序[1000,2000,3000,4000]的顺序进行merge，3个backbone分组测试，任务数量设置为6，其中这6个任务是从这20个任务中随机挑选出来的，任务随机10次取结果的平均值。
# 即先随机出来10组任务，每组的任务数量为6个；然后对这6个任务的参数进行merge（backbone分别分使用resnet 18/50/152这三种），然后计算acc与 loss指标；最后再把这几个值取平均。
import argparse
import copy
import os
import random
from copy import deepcopy

import lightning as L
import numpy as np
import torch
from configs import test_datasets_20
from hydra import compose, initialize
from transformers import AutoConfig, ResNetForImageClassification

from fusion_bench import ResNetForImageClassificationPool
from fusion_bench.constants.clip_vision import TASK_NAMES_TALL20
from fusion_bench.constants.paths import PROJECT_ROOT_PATH
from fusion_bench.method import SimpleAverageAlgorithm
from fusion_bench.taskpool import ResNetForImageClassificationTaskPool
from fusion_bench.utils.json import save_to_json

STEPS = [1000, 2000, 3000, 4000]
BACKS = ["resnet18", "resnet50", "resnet152"]
TASKS20 = copy.deepcopy(TASK_NAMES_TALL20)
N_GROUP = 10
N_TASK = 6
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT_PATH, "outputs")


def load_model(
    backbone: str,
    task_name: str,
    config_name: str,
    step: int,
):
    checkpoints_dir = os.path.join(CHECKPOINT_ROOT, backbone, task_name, config_name)
    raw_checkpoint_path = os.path.join(checkpoints_dir, "raw_checkpoints", "final")
    model_config = AutoConfig.from_pretrained(raw_checkpoint_path)
    raw_model = ResNetForImageClassification(model_config)

    # find checkpoint file
    checkpoint_files = os.listdir(os.path.join(checkpoints_dir, "checkpoints"))
    checkpoint_files = [f for f in checkpoint_files if f"step={step}" in f]
    assert (
        len(checkpoint_files) == 1
    ), f"Expected one checkpoint for step {step}, found {len(checkpoint_files)}"

    # load state dict
    print(f"Loading model for task {task_name} at step {step} from {checkpoints_dir}")
    state_dict_path = os.path.join(checkpoints_dir, "checkpoints", checkpoint_files[0])
    state_dict = torch.load(state_dict_path, map_location="cpu")["state_dict"]
    for k in list(state_dict.keys()):
        assert k.startswith("model."), f"Unexpected key {k} in state dict"
    state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}

    raw_model.load_state_dict(state_dict)
    return raw_model


def merge(models, algo="simple_average"):
    if algo == "simple_average":
        algorithm = SimpleAverageAlgorithm(show_pbar=True, inplace=False)
        merged_model = algorithm.run(models)
        return merged_model
    else:
        raise ValueError(f"Unknown merge algorithm: {algo}")


def eval_model_on_task(model, task: str):
    taskpool = ResNetForImageClassificationTaskPool(
        type="transformers",
        test_datasets={task: test_datasets_20[task]},
        dataloader_kwargs={"batch_size": 64, "num_workers": 4},
        processor_config_path=model.config._name_or_path,
    )
    taskpool._fabric_instance = fabric
    report = taskpool.evaluate(model)
    print(
        f"Evaluation results on task {task}: accuracy={report[task]['accuracy']}, loss={report[task]['loss']}"
    )
    return report[task]["accuracy"], report[task]["loss"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="The backbone model to use.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="batch_size_64_lr_0.001_training_data_ratio_0.5",
        help="The config name for finetuning, e.g., `batch_size_64_lr_0.001_training_data_ratio_0.5`",
    )
    parser.add_argument(
        "--merge-algo",
        type=str,
        default="simple_average",
        help="The model merging algorithm to use.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fabric = L.Fabric(accelerator="cuda", devices=1)

    run_idx = 0
    for step in STEPS:
        for seed in range(N_GROUP):
            random.seed(seed)
            chosen = random.sample(TASKS20, N_TASK)
            models = {
                t: load_model(
                    backbone=args.backbone,
                    task_name=t,
                    config_name=args.config_name,
                    step=step,
                )
                for t in chosen
            }
            # merged_backbone
            print(f"Merging models: {list(models.keys())} at step {step}")
            merged_backbone = merge(
                {n: model.resnet for n, model in models.items()}, algo=args.merge_algo
            )

            # evaluate on chosen tasks
            task_results = {}
            task_acc_list = []
            task_loss_list = []
            for task_name in chosen:
                # substitute backbone
                model = models[task_name]
                model.resnet = deepcopy(merged_backbone)
                task_acc, task_loss = eval_model_on_task(model, task_name)
                task_acc_list.append(task_acc)
                task_loss_list.append(task_loss)
                task_results[task_name] = {
                    "accuracy": task_acc,
                    "loss": task_loss,
                }

            avg_acc = np.mean(task_acc_list)
            avg_loss = np.mean(task_loss_list)

            run_results = {
                "step": step,
                "seed": seed,
                "chosen_tasks": chosen,
                "avg_acc": avg_acc,
                "avg_loss": avg_loss,
                "results": task_results,
            }

            os.makedirs(
                result_dir := f"outputs/exp_step_k/{args.backbone}/{args.config_name}/{args.merge_algo}/step={step}",
                exist_ok=True,
            )
            while os.path.exists(os.path.join(result_dir, f"run_{run_idx}.json")):
                run_idx += 1
            save_to_json(run_results, os.path.join(result_dir, f"run_{run_idx}.json"))
