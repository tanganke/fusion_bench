---
{%- if base_model is not none %}
base_model:
- {{ base_model }}
{%- endif %}
{%- for model in models %}
- {{ model }}
{%- endfor %}
library_name: {{ library_name }}
tags:
{%- for tag in tags %}
- {{ tag }}
{%- endfor %}
---
# {{ title }}

{% if description is not none %}{{ description }}{% endif %}

## Models Merged

This is a merged model created using [fusion-bench](https://github.com/tanganke/fusion_bench).

The following models were included in the merge:

{% if base_model is not none %}
- base model: {{ base_model }}
{%- endif %}
{%- for model in models %}
- {{ model }}
{%- endfor %}

{% if algorithm_config_str is not none or modelpool_config_str is not none %}
## Configuration

The following YAML configuration was used to produce this model:

{% if algorithm_config_str is not none -%}
### Algorithm Configuration

```yaml
{{ algorithm_config_str -}}
```
{%- endif %}

{% if modelpool_config_str is not none -%}
### Model Pool Configuration

```yaml
{{ modelpool_config_str -}}
```
{%- endif %}

{% endif %}
