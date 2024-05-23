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
| **modelpool** | `clip-vit-base-patch32_TA8` | The pool of models to be fused. See [modelpool](../modelpool/README.md) for more information.                                     |
| **method**    | `dummy`                     | The fusion method to be used. See [fusion algorithms](../algorithms/README.md) for more information.                              |
| **taskpool**  | `dummy`                     | The pool of tasks to be evaluated. See [taskpool](../taskpool/README.md) for more information.                                    |
| print_config  | `true`                      | Whether to print the configuration to the console.                                                                                |
| save_report   | `false`                     | the path to save the report. If not specified or is `false`, the report will not be saved. The report will be saved as json file. |
| --cfg, -c     |                             | show the configuration instead of runing.                                                                                         |
| --help, -h    |                             | show this help message and exit.                                                                                                  |

## Basic Examples

merge two CLIP models using task arithmetic:

```bash
fusion_bench method=task_arithmetic \
  modelpool=clip-vit-base-patch32_svhn_and_mnist \
  taskpool=clip-vit-base-patch32_svhn_and_mnist
```

The overall configuration is as follows:

```{.yaml .anotate}
method: # (1)                                                                                                                                                                                                                       
  name: task_arithmetic                                                                                                                                                                                                        
  scaling_factor: 0.5                                                                                                                                                                                                          
modelpool: # (2)                                                                                                                                                                                                                    
   type: huggingface_clip_vision                                                                                                                                                                                                
   models:                                                                                                                                                                                                                      
   - name: _pretrained_                                                                                                                                                                                                         
     path: openai/clip-vit-base-patch32                                                                                                                                                                                         
   - name: svhn                                                                                                                                                                                                                 
     path: tanganke/clip-vit-base-patch32_svhn                                                                                                                                                                                  
   - name: mnist                                                                                                                                                                                                                
     path: tanganke/clip-vit-base-patch32_mnist                                                                                                                                                                                 
taskpool: # (3)                                                                                                                                                                                                                     
  type: clip_vit_classification                                                                                                                                                                                                
  name: clip-vit-base-patch32_svhn_and_mnist                                                                                                                                                                                   
  dataset_type: huggingface_image_classification                                                                                                                                                                               
  tasks:                                                                                                                                                                                                                       
  - name: svhn                                                                                                                                                                                                                 
    dataset:                                                                                                                                                                                                                   
      type: instantiate                                                                                                                                                                                                        
      name: svhn                                                                                                                                                                                                               
      object:                                                                                                                                                                                                                  
        _target_: datasets.load_dataset                                                                                                                                                                                        
        _args_:                                                                                                                                                                                                                
        - svhn                                                                                                                                                                                                                 
        - cropped_digits                                                                                                                                                                                                       
        split: test                                                                                                                                                                                                            
  - name: mnist                                                                                                                                                                                                                
    dataset:                                                                                                                                                                                                                   
      name: mnist                                                                                                                                                                                                              
      split: test                                                                                                                                                                                                              
  clip_model: openai/clip-vit-base-patch32                                                                                                                                                                                     
  batch_size: 128                                                                                                                                                                                                              
  num_workers: 16                                                                                                                                                                                                              
  fast_dev_run: ${fast_dev_run}                                                                                                                                                                                                
fast_dev_run: false                                                                                                                                                                                                            
print_config: true                                                                                                                                                                                                             
save_report: false
```

1. Configuration for method, `fusion_bench.method.load_algorithm_from_config` checks the 'name' attribute of the configuration and returns an instance of the corresponding algorithm.
2. Configuration for model pool, `fusion_bench.modelpool.load_modelpool_from_config` checks the 'type' attribute of the configuration and returns an instance of the corresponding model pool.
3. Configuration for task pool, `fusion_bench.taskpool.load_taskpool_from_config` checks the 'type' attribute of the configuration and returns an instance of the corresponding task pool.


merge multiple CLIP models using simple averaging:

```bash
fusion_bench method=simple_average modelpool=clip-vit-base-patch32_TA8.yaml taskpool=dummy
```


## References

::: fusion_bench.scripts.cli.run_model_fusion
