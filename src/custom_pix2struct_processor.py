# custom_pix2struct.py
import math
import io
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import T5TokenizerFast  # minimal import for tokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

# ----------------------------
# Helper Functions
# ----------------------------

def to_channel_dimension_format(image: np.ndarray, desired_format: str, input_format: Optional[str] = None) -> np.ndarray:
    """
    Minimal helper: if image shape is (H, W, C) and desired_format is "channels_first",
    transpose to (C, H, W). Otherwise, return image unchanged.
    """
    if desired_format == "channels_first":
        if image.ndim == 3 and image.shape[-1] < image.shape[0]:
            return image.transpose(2, 0, 1)
    return image

def get_image_size(image: torch.Tensor, channel_format: str) -> (int, int):
    """Assumes image is a torch tensor in the specified channel format."""
    if channel_format == "channels_first":
        return image.shape[1], image.shape[2]
    else:
        return image.shape[0], image.shape[1]

def to_numpy_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def valid_images(images: List) -> bool:
    return all(isinstance(img, (np.ndarray, Image.Image)) for img in images)

def make_list_of_images(images: Union[Image.Image, np.ndarray, List[Union[Image.Image, np.ndarray]]]) -> List[np.ndarray]:
    if not isinstance(images, list):
        images = [images]
    out = []
    for image in images:
        if isinstance(image, Image.Image):
            out.append(np.array(image))
        elif isinstance(image, np.ndarray):
            out.append(image)
        else:
            raise ValueError("Image must be a PIL.Image or a numpy array.")
    return out

def normalize(image: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (image - mean) / std

def torch_extract_patches(image_tensor: torch.Tensor, patch_height: int, patch_width: int) -> torch.Tensor:
    """
    Extract patches from image_tensor using unfold.
    Returns tensor of shape [1, rows, columns, patch_dim].
    """
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension.
    # Use unfold to extract non-overlapping patches.
    patches = torch.nn.functional.unfold(image_tensor, (patch_height, patch_width), stride=(patch_height, patch_width))
    # patches shape: [batch, patch_dim, num_patches]
    batch_size, patch_dim, num_patches = patches.shape
    # Determine grid dimensions.
    grid_rows = image_tensor.shape[2] // patch_height
    grid_cols = image_tensor.shape[3] // patch_width
    patches = patches.reshape(batch_size, patch_dim, grid_rows, grid_cols)
    patches = patches.permute(0, 2, 3, 1)  # [batch, rows, cols, patch_dim]
    return patches

# ----------------------------
# Custom Patch Extraction Functions
# ----------------------------

def extract_flattened_patches_single(
    image: np.ndarray,
    max_patches: int,
    patch_size: Dict[str, int],
    input_data_format: Optional[Union[str, str]] = None,
) -> np.ndarray:
    """
    Process a single image: resize to limit number of patches, extract patches (non-overlapping),
    prepend row and column IDs to each flattened patch, and pad/truncate to exactly max_patches.
    """
    # Convert image to channels-first using our helper.
    image = to_channel_dimension_format(image, "channels_first", input_data_format)
    image = torch.from_numpy(image.astype(np.float32))
    
    patch_height, patch_width = patch_size["height"], patch_size["width"]
    image_height, image_width = get_image_size(image, "channels_first")
    
    # Compute scaling factor so that the number of patches roughly does not exceed max_patches.
    scale = math.sqrt(max_patches * (patch_height / image_height) * (patch_width / image_width))
    num_feasible_rows = max(min(math.floor(scale * image_height / patch_height), max_patches), 1)
    num_feasible_cols = max(min(math.floor(scale * image_width / patch_width), max_patches), 1)
    resized_height = max(num_feasible_rows * patch_height, 1)
    resized_width = max(num_feasible_cols * patch_width, 1)
    
    image = torch.nn.functional.interpolate(
        image.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).squeeze(0)
    
    patches = torch_extract_patches(image, patch_height, patch_width)
    # patches: [1, rows, cols, depth]
    patches_shape = patches.shape
    rows = patches_shape[1]
    cols = patches_shape[2]
    depth = patches_shape[3]
    
    patches = patches.reshape([rows * cols, depth])
    
    # Generate row and column indices.
    row_ids = torch.arange(rows).reshape(rows, 1).repeat(1, cols).reshape(rows * cols, 1)
    col_ids = torch.arange(cols).reshape(1, cols).repeat(rows, 1).reshape(rows * cols, 1)
    row_ids += 1  # Offset to avoid zeros for padding.
    col_ids += 1
    row_ids = row_ids.to(torch.float32)
    col_ids = col_ids.to(torch.float32)
    
    result = torch.cat([row_ids, col_ids, patches], dim=-1)  # Each patch becomes [row, col, features...]
    
    num_extracted = result.shape[0]
    if num_extracted < max_patches:
        pad_amt = max_patches - num_extracted
        padding = torch.zeros((pad_amt, result.shape[-1]), dtype=result.dtype)
        result = torch.cat([result, padding], dim=0)
    else:
        result = result[:max_patches]
    
    return to_numpy_array(result)

def extract_multi_image_flattened_patches(
    images: List[np.ndarray],
    max_total_patches: int,
    patch_size: Dict[str, int],
    input_data_format: Optional[Union[str, str]] = None,
) -> np.ndarray:
    """
    Process a list of images. Allocate an equal share of patches per image,
    concatenate them, and pad/truncate to exactly max_total_patches.
    """
    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images provided.")
    max_per_image = max_total_patches // num_images
    all_patches = []
    for img in images:
        patches = extract_flattened_patches_single(
            image=img,
            max_patches=max_per_image,
            patch_size=patch_size,
            input_data_format=input_data_format,
        )
        all_patches.append(patches)
    concatenated = np.concatenate(all_patches, axis=0)
    total = concatenated.shape[0]
    feature_dim = concatenated.shape[1]
    if total < max_total_patches:
        pad_amt = max_total_patches - total
        padding = np.zeros((pad_amt, feature_dim), dtype=concatenated.dtype)
        concatenated = np.concatenate([concatenated, padding], axis=0)
    else:
        concatenated = concatenated[:max_total_patches]
    return concatenated

# ----------------------------
# Custom Image Processor
# ----------------------------

class CustomPix2StructImageProcessor:
    """
    A minimal custom image processor for Pix2Struct that:
      - Optionally converts images to RGB
      - Optionally normalizes images
      - Extracts flattened mini-patches from one or more images,
        preserving row and column indices
      - Limits the total number of patches via a max_total_patches parameter
    """
    model_input_names = ["flattened_patches"]

    def __init__(
        self,
        do_convert_rgb: bool = True,
        do_normalize: bool = True,
        patch_size: Optional[Dict[str, int]] = None,
        max_total_patches: int = 2048,
        is_vqa: bool = False,
        **kwargs,
    ) -> None:
        self.do_convert_rgb = do_convert_rgb
        self.do_normalize = do_normalize
        self.patch_size = patch_size if patch_size is not None else {"height": 16, "width": 16}
        self.max_total_patches = max_total_patches
        self.is_vqa = is_vqa

    def extract_flattened_patches(self, images: List[np.ndarray], input_data_format: Optional[Union[str, str]] = None, **kwargs) -> np.ndarray:
        """
        Extract flattened patches from a list of images.
        """
        images = make_list_of_images(images)
        if not valid_images(images):
            raise ValueError("Invalid image type provided.")
        if self.do_convert_rgb:
            images = [np.array(Image.fromarray(img).convert("RGB")) if img.ndim == 3 else img for img in images]
        if self.do_normalize:
            images = [normalize(img, mean=np.mean(img), std=np.std(img)) for img in images]
        flattened = extract_multi_image_flattened_patches(
            images=images,
            max_total_patches=self.max_total_patches,
            patch_size=self.patch_size,
            input_data_format=input_data_format,
        )
        return flattened

    def preprocess(self, images: Union[np.ndarray, List[np.ndarray]], input_data_format: Optional[Union[str, str]] = None, **kwargs):
        """
        Preprocess images and return a dictionary similar to BatchFeature.
        """
        # Ensure images is a list.
        images = make_list_of_images(images)
        flattened_patches = self.extract_flattened_patches(images, input_data_format=input_data_format, **kwargs)
        attention_mask = (flattened_patches.sum(axis=-1) != 0).astype(np.float32)
        return {"flattened_patches": flattened_patches, "attention_mask": attention_mask}

# ----------------------------
# Custom Processor (Image + Tokenizer)
# ----------------------------

class CustomPix2StructProcessor:
    """
    A minimal processor that wraps a custom image processor and a tokenizer.
    """
    def __init__(self, image_processor: CustomPix2StructImageProcessor, tokenizer: T5TokenizerFast) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        # Set tokenizer default: no token_type_ids.
        self.tokenizer.model_max_length = 512

    def __call__(
        self,
        images: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs
    ):
        """
        Process images with the custom image processor and optionally process text.
        Returns a dictionary that merges image and text encodings.
        """
        if images is None and text is None:
            raise ValueError("Provide at least images or text.")

        out = {}
        if images is not None:
            # Process images.
            image_features = self.image_processor.preprocess(images, **kwargs)
            out.update(image_features)
        if text is not None:
            text_features = self.tokenizer(text=text, return_tensors="np", **kwargs)
            # Rename keys to avoid clashing with the image processor's keys.
            if "attention_mask" in text_features:
                text_features["decoder_attention_mask"] = text_features.pop("attention_mask")
            if "input_ids" in text_features:
                text_features["decoder_input_ids"] = text_features.pop("input_ids")
            out.update(text_features)
        return out

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        # Combine input names from tokenizer and image processor.
        t_names = getattr(self.tokenizer, "model_input_names", [])
        i_names = getattr(self.image_processor, "model_input_names", [])
        return list(dict.fromkeys(t_names + i_names))


# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Load a tokenizer (for example, the T5TokenizerFast)
    model_name = "t5-small"  # Replace with your actual model if needed.
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    
    # Initialize our custom image processor.
    image_processor = CustomPix2StructImageProcessor(
        do_convert_rgb=True,
        do_normalize=True,
        patch_size={"height": 16, "width": 16},
        max_total_patches=2048,
        is_vqa=False,
    )
    
    # Create our custom processor.
    processor = CustomPix2StructProcessor(image_processor, tokenizer)
    
    # Example images: load two images using PIL and convert to numpy arrays.
    image1 = np.array(Image.open("path/to/image1.jpg"))
    image2 = np.array(Image.open("path/to/image2.jpg"))
    
    # Process images only.
    encoding = processor(images=[image1, image2])
    print("Encoded image features keys:", list(encoding.keys()))
    print("Flattened patches shape:", encoding["flattened_patches"].shape)
    
    # Process images and text together.
    text = "What is shown in the images?"
    encoding = processor(images=[image1, image2], text=text)
    print("Merged encoding keys:", list(encoding.keys()))
