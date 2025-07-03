"""
This module provides utilities for reading local image files.
"""
import os
from typing import Union
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def read_general2(path, image_root=None) -> str:
    """
    Handle image path with flexible root directory support.
    
    Args:
        path: Image path (can be absolute or relative)
        image_root: Optional root directory to prepend to relative paths
    
    Returns:
        Full path to the image file
    """
    if not path:
        return None
    
    # If path is already absolute, use it directly
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        else:
            print(f"Warning: Absolute path does not exist: {path}")
            return None
    
    # Handle relative paths with image_root
    elif image_root:
        # Combine root directory with relative path
        full_path = os.path.join(image_root, path)
        if os.path.exists(full_path):
            return full_path
        else:
            print(f"Warning: Combined path does not exist: {full_path}")
            return None
    
    # If no image_root provided, try path as is
    else:
        if os.path.exists(path):
            return path
        else:
            print(f"Warning: Path does not exist: {path}")
            return None
