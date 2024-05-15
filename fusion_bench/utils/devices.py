from typing import List, Union


def num_devices(devices: Union[int, List[int], str]) -> int:
    """
    Return the number of devices.

    Args:
        devices: `devices` can be a single int to specify the number of devices, or a list of device ids, e.g. [0, 1, 2, 3]， or a str of device ids, e.g. "0,1,2,3" and "[0, 1, 2]".

    Returns:
        The number of devices.
    """
    if isinstance(devices, int):
        return devices
    elif isinstance(devices, str):
        return len(devices.split(","))
    elif isinstance(devices, list):
        return len(devices)
    else:
        raise TypeError(
            f"devices must be a single int or a list of ints, but got {type(devices)}"
        )
