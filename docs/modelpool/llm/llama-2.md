# LLaMA-2

LLaMA-2 is representing a significant advancement in open-source language modeling. 

## LLaMA-2-7B Models

The LLaMA-2-7B model family provides a good balance between performance and computational efficiency. 
In FusionBench, we offer pre-configured model pools that include various specialized variants for different domains.

This configuration includes the base LLaMA-2-7B model along with specialized variants for chat, mathematics, and coding:

```yaml title="config/modelpool/CausalLMPool/llama-7b_3-models_v1.yaml"
--8<-- "config/modelpool/CausalLMPool/llama-7b_3-models_v1.yaml"
```

### Model Fusion Experiments

#### Simple Average

Merge all models using simple parameter averaging:

```shell
fusion_bench path.log_dir=outputs/llama-2/3-models_v1/simple_average \
    method=linear/simple_average_for_causallm \
    modelpool=CausalLMPool/llama-7b_3-models_v1
```

## Citation

If you use LLaMA-2 models in your research, please cite:

```bibtex
@article{touvron2023llama,
  title={Llama 2: Open foundation and fine-tuned chat models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and Albert, Peter and Almahairi, Amjad and Babaei, Yasmine and Bashlykov, Nikolay and Batra, Soumya and Bhargava, Prajjwal and Bhosale, Shruti and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}
```
