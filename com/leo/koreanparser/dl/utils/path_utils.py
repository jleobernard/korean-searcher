import os

def get_mask_name(filename: str) -> str:
    filename_wihtout_extension, extension = os.path.splitext(filename)
    return f"{filename_wihtout_extension}_mask.{extension}"
