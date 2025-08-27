import logging
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import (
    LightningFabricMixin,
    SimpleProfilerMixin,
    auto_register_config,
)
from fusion_bench.modelpool import CausalLMPool

from .bitdelta_utils.data import get_dataloader, get_dataset
from .bitdelta_utils.diff import compress_diff, save_diff, save_full_model

log = logging.getLogger(__name__)


@auto_register_config
class BitDeltaAlgorithm(
    LightningFabricMixin,
    SimpleProfilerMixin,
    BaseAlgorithm,
):
    def __init__(
        self,
        save_dir: str,
        save_full_model: bool = False,
        lr: float = 1e-4,
        batch_size: int = 4,
        num_steps: int = 100,
        dataset_name: str = "c4",
        subset: str = "en",
        split: str = "train",
        max_length: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def run(self, modelpool: CausalLMPool):
        if self.save_dir is None:
            log.info(
                f"save_dir not set, using log_dir instead. log_dir: {self.log_dir}"
            )
            self.save_dir = self.log_dir

        with self.profile("model loading"):
            tokenizer = modelpool.load_tokenizer()
            base_model = modelpool.load_pretrained_model()
            finetuned_model = modelpool.load_model(modelpool.model_names[0])
            finetuned_compressed_model = modelpool.load_model(modelpool.model_names[0])

        with self.profile("model compression"):
            print(f"compressing diff...")
            compress_diff(base_model, finetuned_model, finetuned_compressed_model)

        # save untrained delta
        save_diff(
            finetuned_compressed_model, os.path.join(self.save_dir, "diff_untrained.pt")
        )

        optimizer = torch.optim.AdamW(
            finetuned_compressed_model.parameters(), lr=self.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.num_steps
        )

        train_num_samples = self.batch_size * self.num_steps
        train_dataset = get_dataset(
            self.dataset_name,
            self.subset,
            "train",
            size=train_num_samples,
        )
        train_dataloader = get_dataloader(
            train_dataset,
            tokenizer,
            self.batch_size,
            num_workers=4,
            max_length=self.max_length,
        )

        bar = tqdm(train_dataloader)

        train_loss_list = []

        # Train loop
        for step, batch in enumerate(bar):
            batch1 = {k: v.to(finetuned_model.device) for k, v in batch.items()}
            with torch.inference_mode():
                finetuned_outputs = finetuned_model(**batch1)

            batch2 = {
                k: v.to(finetuned_compressed_model.device) for k, v in batch.items()
            }
            finetuned_compressed_outputs = finetuned_compressed_model(**batch2)

            loss = F.mse_loss(
                finetuned_outputs.logits.clone().to(
                    finetuned_compressed_outputs.logits.device
                ),
                finetuned_compressed_outputs.logits,
            )

            train_loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            bar.set_description(f"train loss: {loss.item()}")

        # save trained delta
        save_diff(finetuned_compressed_model, os.path.join(self.save_dir, "diff.pt"))

        if self.save_full_model:
            print("saving uncalibrated model")
            save_full_model(
                base_model,
                tokenizer,
                os.path.join(self.save_dir, "diff_untrained.pt"),
                os.path.join(self.save_dir, "uncalibrated_model"),
                device="cpu",
            )
            print("saving calibrated model")
            save_full_model(
                base_model,
                tokenizer,
                os.path.join(self.save_dir, "diff.pt"),
                os.path.join(self.save_dir, "calibrated_model"),
                device="cpu",
            )

        del base_model, finetuned_model, finetuned_compressed_model
        torch.cuda.empty_cache()
