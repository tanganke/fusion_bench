---
name: Bug Report
about: Report a bug to help us improve FusionBench
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## Describe the Bug
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:

1. **Command or Script Execution:**
   ```bash
   # Paste your command here
   fusion_bench method=... modelpool=... taskpool=...
   ```

2. **Configuration Used:**
   ```yaml
   # Paste relevant configuration here
   ```

3. **Error Message or Log:**
   ```
   Please include the full stack trace, especially the location of the error.
   If launched from fusion_bench CLI, set the environment variable HYDRA_FULL_ERROR=1
   to obtain the full stack trace:
   
   HYDRA_FULL_ERROR=1 fusion_bench method=...
   ```

## Expected Behavior
A clear and concise description of what you expected to happen.

## Environment Information
Please provide the following information:

- **Operating System:** [e.g., Ubuntu 22.04, macOS 14.0, Windows 11]
- **Python Version:** [e.g., 3.10.12]
- **FusionBench Version:** [e.g., 0.2.29, or git commit hash]
- **PyTorch Version:** [e.g., 2.1.0]
- **CUDA Version (if applicable):** [e.g., 11.8]
- **Other Dependencies:** 
  ```bash
  # Run: pip list | grep -E "transformers|datasets|hydra-core|lightning"
  ```

## Screenshots or Logs
If applicable, add screenshots or additional logs to help explain your problem.

## Additional Context
Add any other context about the problem here:
- Does this happen consistently or intermittently?
- Have you made any modifications to the code?
- Any other relevant information
