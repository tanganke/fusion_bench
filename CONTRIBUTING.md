# Contributing to Fusion Bench

> [!NOTE]
> This guideline is very much a work in progress. We are working on improving it and making it more comprehensive. If you have any suggestions or feedback, please let us know.

Thank you for considering contributing to Fusion Bench! We welcome contributions from the community and are excited to collaborate with you.

## Understanding the Configuration System

FusionBench uses [Hydra](https://hydra.cc/) for configuration management. If you're planning to contribute:

- Familiarize yourself with [Hydra's documentation](https://hydra.cc/docs/intro/)
- Review the [config/README.md](./config/README.md) for FusionBench-specific configuration patterns
- Look at existing configuration files in the `config/` directory for examples
- Ensure your YAML configurations follow the established patterns (e.g., using `_target_` for class instantiation)

## Code Style

Currently, we are using the `black` code formatter and `isort` to format our code. We recommend you to use these tools to format your code before submitting a pull request.

## Reporting Issues

If you encounter any bugs or have suggestions for improvements, please open an issue on our GitHub repository. Provide as much detail as possible to help us understand and address the issue.
