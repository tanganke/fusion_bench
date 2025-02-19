"""
The function to train the model
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from accelerate.accelerator import Accelerator
from accelerate.data_loader import DataLoaderShard
from accelerate.logging import MultiProcessAdapter
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    SwitchTransformersForConditionalGeneration,
)


@dataclass
class TrainingArguments:
    overrode_max_train_steps: bool
    output_dir: str
    per_device_train_batch_size: int
    num_epochs: Optional[int]
    max_train_steps: Optional[int]
    checkpoint_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = 1
    num_eval_steps: Optional[int] = None
    eval_at_the_beginning: bool = False
    log_steps: Optional[int] = 100
    no_eval_until_epochs: Optional[int] = 0


def accelerate_run_train(
    accelerator: Accelerator,
    model: PreTrainedModel,
    optimizer: AcceleratedOptimizer,
    lr_scheduler: AcceleratedScheduler,
    training_args: TrainingArguments,
    train_dataloader: DataLoaderShard,
    logger: MultiProcessAdapter,
    eval_dataloader: Optional[DataLoaderShard] = None,
    wandb=None,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    evaluate_fn=None,
    evaluate_loss_key="loss",
    print_outputs=False,
):
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    no_eval_until_steps = (
        training_args.no_eval_until_epochs * num_update_steps_per_epoch
    )
    if training_args.num_eval_steps is None:
        #  Eval at the end of each epoch
        training_args.num_eval_steps = num_update_steps_per_epoch
    if training_args.overrode_max_train_steps:
        training_args.max_train_steps = (
            training_args.num_epochs * num_update_steps_per_epoch
        )
    if training_args.checkpoint_steps is None:
        # Default to never saving checkpoints
        training_args.checkpoint_steps = training_args.max_train_steps + 2
    training_args.num_epochs = math.ceil(
        training_args.max_train_steps / num_update_steps_per_epoch
    )
    per_device_train_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    total_batch_size = (
        (
            per_device_train_batch_size
            * accelerator.num_processes
            * gradient_accumulation_steps
        )
        if accelerator.num_processes is not None
        else (per_device_train_batch_size * gradient_accumulation_steps)
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {training_args.num_epochs}")
    logger.info(f"  Num Samples = {len(train_dataloader)}")
    logger.info(
        f"  Instantaneous batch size per device = {per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")

    progress_bar = tqdm(
        range(training_args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    best_eval = float("inf")

    # Train!
    for epoch in range(training_args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                if print_outputs and accelerator.is_local_main_process:
                    print(outputs)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if (
                accelerator.is_local_main_process
                and completed_steps % training_args.log_steps == 0
            ):
                logger.info(f"epoch {epoch}, step {step}: loss {loss.item()}")
                if wandb is not None:
                    wandb.log(
                        {
                            "train_loss": loss.item(),
                            "epoch": completed_steps / num_update_steps_per_epoch,
                            "learning_rate": lr_scheduler.get_lr(),
                        },
                        step=completed_steps,
                    )
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
            if (
                completed_steps % training_args.checkpoint_steps == 0
                and completed_steps > 0
            ):
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(
                        training_args.output_dir,
                        "checkpoint-{}".format(completed_steps),
                    ),
                    is_main_process=accelerator.is_local_main_process,
                    save_function=accelerator.save,
                )
            if (
                completed_steps % training_args.num_eval_steps == 0
                and completed_steps >= no_eval_until_steps
                and accelerator.sync_gradients
            ) or (step == 0 and training_args.eval_at_the_beginning):
                if eval_dataloader is not None:
                    model.eval()
                    losses = []
                    output_labels = []
                    output_predictions = []
                    for eval_step, eval_batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**eval_batch)
                        if evaluate_fn is not None:
                            eval_labels = accelerator.gather(eval_batch["labels"])
                            output_labels += torch.cat(
                                [
                                    eval_labels,
                                    torch.ones(
                                        eval_labels.shape[0],
                                        tokenizer.model_max_length
                                        - eval_labels.shape[1],
                                        dtype=eval_labels.dtype,
                                        device=eval_labels.device,
                                    )
                                    * -100,
                                ],
                                dim=-1,
                            )
                            eval_logits = accelerator.gather(outputs.logits)
                            output_predictions += eval_logits.argmax(dim=-1).tolist()
                        loss = outputs[evaluate_loss_key]
                        losses.append(accelerator.gather_for_metrics(loss))
                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    if evaluate_fn is not None:
                        output_labels = torch.stack(output_labels, dim=0)
                        eval_res = evaluate_fn(
                            predictions=output_predictions, labels=output_labels
                        )
                    else:
                        eval_res = dict()
                    metric_key = (
                        list(eval_res.keys())[0]
                        if len(eval_res) > 0
                        else evaluate_loss_key
                    )
                    eval_res[evaluate_loss_key] = eval_loss.item()
                    try:
                        perplexity = math.exp(eval_res[evaluate_loss_key])
                    except OverflowError:
                        perplexity = float("inf")
                    eval_res["perplexity"] = perplexity

                    is_language_modeling = metric_key == evaluate_loss_key

                    if (is_language_modeling and eval_res[metric_key] < best_eval) or (
                        not is_language_modeling and eval_res[metric_key] > -best_eval
                    ):  # best_eval is inited as inf
                        best_eval = (
                            eval_res[metric_key]
                            if is_language_modeling
                            else -eval_res[metric_key]
                        )
                        wandb.summary["best_" + metric_key] = eval_res[metric_key]
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            os.path.join(training_args.output_dir, "best"),
                            is_main_process=accelerator.is_local_main_process,
                            save_function=accelerator.save,
                        )
                        if accelerator.is_local_main_process:
                            tokenizer.save_pretrained(
                                os.path.join(training_args.output_dir, "best")
                            )
                            print("Best model saved")

                    if accelerator.is_local_main_process:
                        print(
                            f"epoch {epoch} step {completed_steps}: eval loss {eval_res[evaluate_loss_key]}, perplexity {perplexity}"
                        )
                        if wandb is not None:
                            eval_res = {("eval_" + k): v for k, v in eval_res.items()}
                            wandb.log(eval_res, step=completed_steps)

            if completed_steps >= training_args.max_train_steps:
                break

    # Finish Training!
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-{}".format(completed_steps)),
        is_main_process=accelerator.is_local_main_process,
        save_function=accelerator.save,
    )
    tokenizer.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-{}".format(completed_steps))
    )

    if accelerator.is_local_main_process and wandb is not None:
        wandb.finish()


def freeze_switch_routers_for_finetuning(
    model: SwitchTransformersForConditionalGeneration,
) -> SwitchTransformersForConditionalGeneration:
    model.router_z_loss_coef = 0
    model.router_aux_loss_coef = 0
    for name, param in model.named_parameters():
        if "router" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    return model
