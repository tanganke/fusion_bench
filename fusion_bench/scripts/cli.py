import importlib
import importlib.resources
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from rich import print as rich_print
from rich.syntax import Syntax

from ..modelpool import load_modelpool
from ..method import load_algorithm


@hydra.main(
    config_path=os.path.join(
        importlib.import_module("fusion_bench").__path__[0], "../config"
    ),
    config_name="example_config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if cfg.print_config:
        rich_print(
            Syntax(
                OmegaConf.to_yaml(cfg),
                "yaml",
                tab_size=2,
                line_numbers=True,
            )
        )

    modelpool = load_modelpool(cfg.modelpool)
    algorithm = load_algorithm(cfg.method)
    merged_model = algorithm.fuse(modelpool)


if __name__ == "__main__":
    main()
