Here is a table of supported algorithms in the benchmark:

=== "Model Merging Algorithms"

    | Algorithm Name                        | Class Path                                                                                           |
    | ------------------------------------- | ---------------------------------------------------------------------------------------------------- |
    | `dummy`                               |
    | `clip_finetune`                       | `.classification.clip_finetune.ImageClassificationFineTuningForCLIP`                                 |
    | `TaskVectorCosSimilarity`             | `.analysis.task_vector_cos_similarity.TaskVectorCosSimilarity`                                       |
    | `simple_ensemble`                     | `.ensemble.EnsembleAlgorithm`                                                                        |
    | `weighted_ensemble`                   | `.ensemble.WeightedEnsembleAlgorithm`                                                                |
    | `max_model_predictor`                 | `.ensemble.MaxModelPredictorAlgorithm`                                                               |
    | `simple_average`                      | `.simple_average.SimpleAverageAlgorithm`                                                             |
    | `weighted_average`                    | `.weighted_average.weighted_average.WeightedAverageAlgorithm`                                        |
    | `weighted_average_for_llama`          | `.weighted_average.llama.WeightedAverageForLLama`                                                    |
    | `clip_fisher_merging`                 | `.fisher_merging.clip_fisher_merging.FisherMergingAlgorithmForCLIP`                                  |
    | `gpt2_fisher_merging`                 | `.fisher_merging.gpt2_fisher_merging.FisherMergingAlgorithmForGPT2`                                  |
    | `clip_regmean`                        | `.regmean.clip_regmean.RegMeanAlgorithmForCLIP`                                                      |
    | `gpt2_regmean`                        | `.regmean.gpt2_regmean.RegMeanAlgorithmForGPT2`                                                      |
    | `task_arithmetic`                     | `.task_arithmetic.TaskArithmeticAlgorithm`                                                           |
    | `ties_merging`                        | `.ties_merging.ties_merging.TiesMergingAlgorithm`                                                    |
    | `clip_task_wise_adamerging`           | `.adamerging.clip_task_wise_adamerging.CLIPTaskWiseAdaMergingAlgorithm`                              |
    | `clip_layer_wise_adamerging`          | `.adamerging.clip_layer_wise_adamerging.CLIPLayerWiseAdaMergingAlgorithm`                            |
    | `singular_projection_merging`         | `fusion_bench.method.smile_upscaling.singular_projection_merging.SingularProjectionMergingAlgorithm` |
    | `pwe_moe_ls_for_clip`                 | `.pwe_moe.clip_pwe_moe.PWEMoELinearScalarizationForCLIP`                                             |
    | `pwe_moe_epo_for_clip`                | `.pwe_moe.clip_pwe_moe.PWEMoExactParetoOptimalForCLIP`                                               |
    | `clip_concrete_task_arithmetic`       | `.concrete_subspace.clip_concrete_task_arithmetic.ConcreteTaskArithmeticAlgorithmForCLIP`            |
    | `clip_concrete_task_wise_adamerging`  | `.concrete_subspace.clip_concrete_adamerging.ConcreteTaskWiseAdaMergingForCLIP`                      |
    | `clip_concrete_layer_wise_adamerging` | `.concrete_subspace.clip_concrete_adamerging.ConcreteLayerWiseAdaMergingForCLIP`                     |
    | `mixtral_moe_merging`                 | `.mixture_of_experts.mixtral_merging.MixtralMoEMergingAlgorithm`                                     |
    | `mixtral_for_causal_lm_merging`       | `.mixture_of_experts.mixtral_merging.MixtralForCausalLMMergingAlgorithm`                             |
    | `clip_weight_ensembling_moe`          | `.we_moe.clip_we_moe.CLIPWeightEnsemblingMoEAlgorithm`                                               |

=== "Model Mixing Algorithms"

    | Algorithm Name  | Class                                                                                                                                                                        | Description                                                                                                        |
    | --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
    | Depth Upscaling | [DepthUpscalingAlgorithm][fusion_bench.method.DepthUpscalingAlgorithm]                                                                                                       | Depth upscaling algorithm that concatenates the layers of model. [:simple-arxiv:](http://arxiv.org/abs/2312.15166) |
    | MoE Upscaling   | [MixtralUpscalingAlgorithm][fusion_bench.method.MixtralUpscalingAlgorithm], [MixtralForCausalLMUpscalingAlgorithm][fusion_bench.method.MixtralForCausalLMUpscalingAlgorithm] | Mixture of Experts upscaling algorithm that merges the models. [:simple-arxiv:](http://arxiv.org/abs/2212.05055)   |
    | SMILE Upscaling | [SmileUpscalingAlgorithm][fusion_bench.method.SmileUpscalingAlgorithm]                                                                                                       | Upscale models to sparse low-rank MoE model. [:simple-arxiv:](https://arxiv.org/abs/2408.10174)                    |

=== "Model Ensemble Algorithms"

    | Algorithm Name    | Class                                                                      | Description                                                                             |
    | ----------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
    | Simple Ensemble   | [SimpleEnsembleAlgorithm][fusion_bench.method.SimpleEnsembleAlgorithm]                 | Simple ensemble algorithm that averages the predictions of multiple models.             |
    | Weighted Ensemble | [WeightedEnsembleAlgorithm][fusion_bench.method.WeightedEnsembleAlgorithm] | Ensemble algorithm that averages the predictions of multiple models with given weights. |

=== "Others"

    These algorithms are not directly related to model fusion, but they are used in the benchmark for other purposes.

    | Algorithm Name  | Class                                                | Description            |
    | --------------- | ---------------------------------------------------- | ---------------------- |
    | Dummy Algorithm | [DummyAlgorithm][fusion_bench.method.DummyAlgorithm] | Return model as it is. |


You can find the implementation of these algorithms in the corresponding files.
