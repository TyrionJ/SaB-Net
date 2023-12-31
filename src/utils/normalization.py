import numpy as np


def CTNormalize(image: np.ndarray, foreground_properties: dict) -> np.ndarray:
    mean_intensity = foreground_properties['mean']
    std_intensity = foreground_properties['std']
    lower = foreground_properties['percentile_00_5']
    upper = foreground_properties['percentile_99_5']

    image = image.clip(lower, upper)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)

    return image
