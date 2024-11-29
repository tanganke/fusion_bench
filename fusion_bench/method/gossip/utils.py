import torch


def get_memory_usage(desc):
    """
    obtain the current GPU memory usage

    Returns:
        str: A string containing the allocated and cached memory in MB.
    """
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    cached = torch.cuda.memory_reserved() / 1024**2  # 转换为 MB
    return (
        f"{desc}\nAllocated Memory: {allocated:.2f} MB\nCached Memory: {cached:.2f} MB"
    )
