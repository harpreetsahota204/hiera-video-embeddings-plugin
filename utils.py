import numpy as np
import torch
import torch.nn.functional as F

CHECKPOINTS = {
    "pt_checkpoint": "mae_k400",
    "ft_checkpoint": "mae_k400_ft_k400"
}

HIERA_MODELS = {
    "Hiera-B": "hiera_base_16x224",
    "Hiera-B+": "hiera_base_plus_16x224",
    "Hiera-L": "hiera_large_16x224",
    "Hiera-H": "hiera_huge_16x224"
}

def get_device():
    """Helper function to determine the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) device")
    else:
        device = "cpu"
        print("Using CPU device")
    return device

def load_video_frames(video_data):
    """
    Load video data into a list of numpy arrays.

    Args:
        video_data: Data compatible with FFmpegVideoReader.read()

    Returns:
        torch.Tensor: Video tensor with shape [T, H, W, C]
    """
    frames = [np.array(frame) for frame in video_data]
    return torch.from_numpy(np.stack(frames)).float()

def prepare_tensor_dimensions(frames):
    """
    Add batch dimension and rearrange to model-expected format.

    Args:
        frames (torch.Tensor): Video tensor with shape [T, H, W, C]

    Returns:
        torch.Tensor: Rearranged tensor with shape [1, C, T, H, W]
    """
    frames = frames.unsqueeze(0)  # Add batch: [1, T, H, W, C]
    return frames.permute(0, 4, 1, 2, 3).contiguous()  # To: [1, C, T, H, W]

def resize_video_tensor(frames, num_frames=16, height=224, width=224):
    """
    Resize video tensor to specified dimensions.

    Args:
        frames (torch.Tensor): Video tensor with shape [1, C, T, H, W]
        num_frames (int): Target number of frames
        height (int): Target height
        width (int): Target width

    Returns:
        torch.Tensor: Resized tensor with shape [1, C, num_frames, height, width]
    """
    return F.interpolate(frames, size=(num_frames, height, width), mode="trilinear")

def process_video_data(video_data):
    """
    Process video data and convert it to a tensor formatted for Hiera model input.
    
    Args:
        video_data: Data compatible with FFmpegVideoReader.read()

    Returns:
        torch.Tensor: Video tensor with shape (1, 3, 16, 224, 224) where:
            - 1: Batch dimension
            - 3: RGB channels
            - 16: Number of frames (temporal dimension)
            - 224, 224: Height and width of each frame

    Note:
        The output tensor is normalized to float values but not normalized 
        to specific mean/std values. Apply normalization if required by model.
    """
    frames = load_video_frames(video_data)
    frames = prepare_tensor_dimensions(frames)
    frames = resize_video_tensor(frames)
    return frames