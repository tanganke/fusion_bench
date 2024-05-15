# fusion_bench

`fusion_bench` is the command line interface for running the benchmark.
It takes a configuration file as input, which specifies the models, fusion method to be used, and the datasets to be evaluated. 

```
fusion_bench [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...
```

`fusion_bench` has the following options:

| **Option**   | **Default**               | **Description**                                    |
| ------------ | ------------------------- | -------------------------------------------------- |
| method       | `simple_average`          | The fusion method to be used.                      |
| modelpool    | `huggingface_clip_vision` | The pool of models to be fused.                    |
| print_config | `true`                    | Whether to print the configuration to the console. |

## Basic Examples

merge multiple CLIP models using simple averaging:

```bash
fusion_bench method=simple_average modelpool=clip-vit-base-patch32_TA8.yaml
```
