import os

import numpy as np
import torch

import fiftyone as fo
from fiftyone.core.models import Model
from fiftyone.core.utils import add_sys_path

from typing import List

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from utils import (
        process_video_data,
        get_device,
    )

class HieraVideoEmbeddingModel(Model):
    """A model for extracting embeddings from videos using Hiera models.

    Args:
        model_name (str): Name of the pretrained Hiera model to use
        checkpoint (str): Model checkpoint path or identifier
        embedding_type (str): Type of embedding to extract ('terminal' or 'hierarchical')
        normalize (bool, optional): Whether to L2 normalize the embeddings. Defaults to False.

    Attributes:
        model_name (str): Name of the loaded pretrained model
        embedding_type (str): Type of embedding being extracted
        model (torch.nn.Module): The loaded Hiera model instance
        checkpoint (str): The model checkpoint identifier
        device (torch.device): Device used for computation (CPU/GPU)
        normalize (bool): Whether to normalize embeddings
    """
    def __init__(
        self, 
        model_name: str, 
        checkpoint: str, 
        embedding_type: str,
        normalize: bool
    ):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.embedding_type = embedding_type
        self.normalize = normalize

        # Validate embedding type
        valid_types = ["terminal", "hierarchical"]
        if self.embedding_type not in valid_types:
            raise ValueError(f"Invalid embedding type: {embedding_type}. Must be one of {valid_types}")

        self.model = torch.hub.load(
            "facebookresearch/hiera",
            model=self.model_name,
            checkpoint=self.checkpoint,
            pretrained=True,
            )

        # Set up device
        self.device = get_device()

        # Move model to appropriate device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    @property
    def media_type(self):
        return "video"

    def extract_embeddings(self, intermediates: List[torch.Tensor]) -> np.ndarray:
        """Extract embeddings from model intermediate tensors.

        Args:
            intermediates (List[torch.Tensor]): List of intermediate feature tensors 
                from each layer of the model

        Returns:
            np.ndarray: Extracted embeddings with shape:
                - (768,) for terminal embeddings (final layer only)
                - (1440,) for hierarchical embeddings (concatenated across layers)
                If normalize=True, embeddings will be normalized.
        """
        if self.embedding_type == "terminal":
            # Terminal embedding from final layer [1, 768]
            last_layer = intermediates[-1]
            emb = last_layer.mean(dim=[1,2,3])
            if self.normalize:
                emb = self.model.norm(emb).detach()  # Add detach() after normalization
            terminal_embedding = emb.cpu().numpy()
            return terminal_embedding.squeeze()

        if self.embedding_type == "hierarchical":
            # Hierarchical embedding from all layers [1, 96+192+384+768]
            layer_embeddings = []
            for tensor in intermediates:
                emb = tensor.mean(dim=[1,2,3])
                if self.normalize:
                    emb = self.model.norm(emb).detach()  # Add detach() after normalization
                layer_embeddings.append(emb)
            hierarchical_embedding = torch.cat(layer_embeddings, dim=1).cpu().numpy()
            return hierarchical_embedding.squeeze()

    def _predict(self, frames: torch.Tensor) -> np.ndarray:
        """Process a video tensor and extract embeddings.

        Args:
            frames (torch.Tensor): Video tensor of shape (1, 3, 16, 224, 224)
                where:
                - 1: batch size
                - 3: RGB channels
                - 16: number of frames
                - 224, 224: frame height and width

        Returns:
            np.ndarray: Embedding array with shape:
                - (768,) for terminal embeddings
                - (1440,) for hierarchical embeddings
        """
        frames = frames.to(self.device)

        with torch.no_grad():
            _ , intermediates = self.model(frames, return_intermediates=True)

        return self.extract_embeddings(intermediates)

    def predict(self, args: np.ndarray) -> np.ndarray:
        """Extract embeddings from a video array.

        Args:
            args (np.ndarray): Video array of shape (T, H, W, C) where:
                - T: number of frames
                - H: frame height
                - W: frame width
                - C: number of channels (3 for RGB)

        Returns:
            np.ndarray: Embedding array with shape:
                - (768,) for terminal embeddings
                - (1440,) for hierarchical embeddings
        """
        frames = process_video_data(args) #frames come out as tensors after processing
        predictions = self._predict(frames) #tensors go into _predict and come out as ndarray
        return predictions

    def predict_all(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """Extract embeddings from multiple videos.

        Args:
            videos (List[np.ndarray]): List of video arrays, each with 
                shape (T, H, W, C)

        Returns:
            List[np.ndarray]: List of embedding arrays for each video
        """
        return [self.predict(video) for video in videos]

def run_embeddings_model(
    dataset: fo.Dataset,
    model_name: str,
    checkpoint: str,
    emb_field: str,
    embedding_types: str,
    normalize: bool
) -> None:
    """Run the Hiera embedding model on a FiftyOne dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to process
        model_name (str): Name of the Hiera model to use
        checkpoint (str): Model checkpoint identifier
        emb_field (str): Name of the field to store the extracted embeddings
        embedding_types (str): Type of embeddings to extract:
            - 'terminal': final layer embeddings (768-dim)
            - 'hierarchical': multi-scale embeddings (1440-dim)
        normalize (bool, optional): Whether to L2 normalize the embeddings. Defaults to False.
    """
    model = HieraVideoEmbeddingModel(
        model_name, 
        checkpoint, 
        embedding_types,
        normalize=normalize
        )

    dataset.apply_model(
        model, 
        label_field=emb_field
        )