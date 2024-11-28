import hydra.core.hydra_config


def get_hydra_output_dir():
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    return hydra_cfg.runtime.output_dir


def config_priority_get(priority_config, general_config, key, default):
    """
    Retrieve a configuration value with priority.

    This function retrieves the value associated with `key` from `priority_config` if it exists.
    If the key is not found in `priority_config`, it retrieves the value from `general_config`.
    If the key is not found in either configuration, it returns the provided `default` value.

    Args:
        priority_config (dict): The configuration dictionary with higher priority.
        general_config (dict): The general configuration dictionary.
        key (str): The key to look up in the configuration dictionaries.
        default: The default value to return if the key is not found in either configuration.

    Returns:
        The value associated with `key` from `priority_config` or `general_config`, or the `default` value if the key is not found.
    """
    if key in priority_config:
        return priority_config[key]
    return general_config.get(key, default)
