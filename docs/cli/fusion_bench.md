# fusion_bench

## Details and Options

`fusion_bench` is the command line interface for running the benchmark.
It takes a configuration file as input, which specifies the models, fusion method to be used, and the datasets to be evaluated. 

```
fusion_bench [--config-path CONFIG_PATH] [--config-name CONFIG_NAME] \
    OPTION_1=VALUE_1 OPTION_2=VALUE_2 ...
```

`fusion_bench` has the following options, `method`, `modelpool`, and `taskpool` are the most important ones ammong these options:

| **Option**    | **Default**                 | **Description**                                                                                                                   |
| ------------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **modelpool** | `clip-vit-base-patch32_TA8` | The pool of models to be fused. See [modelpool](../modelpool/README.md) for more information.                                      |
| **method**    | `dummy`                     | The fusion method to be used. See [fusion algorithms](../algorithms/README.md) for more information.                               |
| **taskpool**  | `dummy`                     | The pool of tasks to be evaluated. See [taskpool](../taskpool/README.md) for more information.                                     |
| print_config  | `true`                      | Whether to print the configuration to the console.                                                                                |
| save_report   | `false`                     | the path to save the report. If not specified or is `false`, the report will not be saved. The report will be saved as json file. |

## Basic Examples

merge multiple CLIP models using simple averaging:

```bash
fusion_bench method=simple_average modelpool=clip-vit-base-patch32_TA8.yaml taskpool=dummy
```


## Options


## References

::: fusion_bench.scripts.cli.run_model_fusion
