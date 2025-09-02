---
title: 📚 Documentation
status: new
---
# 📚 Contributing to Documentation

We welcome contributions to the FusionBench documentation! This guide will help you understand how to contribute effectively to our documentation built with [MkDocs](https://www.mkdocs.org/), [Material theme](https://squidfunk.github.io/mkdocs-material/), and [mkdocstrings](https://mkdocstrings.github.io/python/).

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or later
- Git

### Setting Up Local Development

1. Clone the repository:

    ```bash
    git clone https://github.com/tanganke/fusion_bench.git
    cd fusion_bench
    ```

2. Install documentation dependencies:

    ```bash
    pip install -e ".[docs]"
    ```

3. Serve the documentation locally:

    ```bash
    mkdocs serve
    # or add `--dirty` to only rebuild changed files
    mkdocs serve --dirty
    ```

    The documentation will be available at `http://localhost:8000` with live reload enabled.

    > For external access (useful for containers or remote development):
    >
    > ```bash
    > mkdocs serve -a 0.0.0.0:8000
    > ```

## 📁 Documentation Structure

```text
docs/
├── README.md      # Home page
├── algorithms/    # Algorithm documentation
├── api/           # API reference (auto-generated)
├── cli/           # CLI documentation
├── guides/        # User guides and tutorials
├── modelpool/     # Model pool documentation
├── taskpool/      # Task pool documentation
├── css/           # Custom stylesheets
├── javascripts/   # Custom JavaScript
└── images/        # Documentation images
```

## ✍️ Writing Guidelines

### Markdown Standards

- Use **ATX-style headers** (`#`, `##`, `###`, etc.)
- Use **code fences** with language specification:

    ```python
    def example_function():
        return "Hello, World!"
    ```

- Use [**admonitions**](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) for important notes:

=== "Examples"

    !!! note "Important"
        This is an important note.
  
    !!! warning "Caution"
        This requires careful attention.
  
    !!! tip "Pro Tip"
        This is a helpful tip.

=== "Source"

    ```markdown
    !!! note "Important"
        This is an important note.
    
    !!! warning "Caution"
        This requires careful attention.
    
    !!! tip "Pro Tip"
        This is a helpful tip.
    ```

### API Documentation

Our API documentation is auto-generated using **mkdocstrings**. To document code:

1. Write comprehensive docstrings:

    ```python
    def example_function(param1: str, param2: int = 10) -> str:
        """
        Brief description of the function.
        
        Args:
            param1: Description of the first parameter.
            param2: Description of the second parameter. Defaults to 10.
        
        Returns:
            Description of the return value.
        
        Example:
            ```python
            result = example_function("hello", 5)
            print(result)  # Output: "hello5"
            ```
        """
        return param1 + str(param2)
    ```

2. Add API pages in `docs/api/` directory:

    ```markdown
    # Module Name
    
    Brief description of the module.
    
    ::: fusion_bench.module_name
    ```

### Mathematical Expressions

Use **MathJax** for mathematical notation:

=== "Examples"

    Inline math: $E = mc^2$

    Block math:
    
    $$\frac{\partial L}{\partial w} = \sum_{i=1}^{n} (y_i - \hat{y}_i) x_i$$

=== "Source"

    ```markdown
    Inline math: $E = mc^2$

    Block math:
    
    $$\frac{\partial L}{\partial w} = \sum_{i=1}^{n} (y_i - \hat{y}_i) x_i$$
    ```

> Thank you for contributing to FusionBench documentation! 🚀
