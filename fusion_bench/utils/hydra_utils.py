import hydra.core.hydra_config


def get_hydra_output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg.runtime.output_dir
