import os

import numpy as np
import torch

import fiftyone as fo
from fiftyone.core.models import Model
from fiftyone.core.utils import add_sys_path

from importlib.util import find_spec
from typing import List, Dict


with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from utils import (
        read_video_from_path,
        get_device,
    )

class HieraVideoEmbeddingModel(Model):
    """
    HieraVideoEmbeddingModel extracts embeddings from videos using one of the Hiera models.

    Attributes:
        model_name (str): Name of the pretrained model to use
        embedding_type (str): Type of embedding to extract ('terminal' or 'hierarchical')  # Fixed attribute name
        model (torch.nn.Module): The loaded Hiera model  # Added type
        checkpoint (str): The model checkpoint (pretrained or finetuned on K400)
        device (torch.device): The device used for computation  # Added missing attribute
    """
    def __init__(self, model_name: str, checkpoint: str, embedding_type: str):
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.embedding_type = embedding_type

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
        """
        Extracts specified embedding type(s) from model intermediates.

        Args:
            intermediates (List[torch.Tensor]): List of intermediate tensors from model

        Returns:
            np.ndarray: numpy array containing requested embedding types:
                        - 'terminal': Final layer embedding [768]
                        - 'hierarchical': Concatenated multi-scale embedding [1440]
        """

        if self.embedding_type == "terminal":
            # Terminal embedding from final layer [1, 768]
            last_layer = intermediates[-1]
            terminal_embedding = last_layer.mean(dim=[1,2,3]).cpu().numpy()
            terminal_embedding = terminal_embedding.squeeze()
            return terminal_embedding

        if self.embedding_type == "hierarchical":
            # Hierarchical embedding from all layers [1, 96+192+384+768]
            layer_embeddings = []
            for tensor in intermediates:
                emb = tensor.mean(dim=[1,2,3])
                layer_embeddings.append(emb)
            hierarchical_embedding = torch.cat(layer_embeddings, dim=1).cpu().numpy()
            hierarchical_embedding = hierarchical_embedding.squeeze()
            return hierarchical_embedding

    def _predict(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs embedding extraction on a video tensor.  # Fixed description

        Args:
            frames (torch.Tensor): Video tensor with shape (1, 3, 16, 224, 224)

        Returns:
            np.ndarray: Embedding array with shape:
                       - (768,) for terminal embeddings
                       - (1440,) for hierarchical embeddings
        """
        frames = frames.to(self.device)

        with torch.no_grad():
            _ , intermediates = self.model(frames, return_intermediates=True)

        return self.extract_embeddings(intermediates)

    def predict(self, args) -> np.ndarray:
        """
        Predicts embeddings for the given image.

        Args:
            args (np.ndarray): The input image as a numpy array

        Returns:
            np.ndarray: numpy array containing requested embedding types:
        """
        frames = read_video_from_path(args)
        predictions = self._predict(frames)
        return predictions

    def _predict_all(self, videos: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Performs prediction on a list of video tensors.

        Args:
            videos (List[torch.Tensor]): List of video tensors 

        Returns:
            List[np.ndarray]: List of embedding arrays for each video
        """
        return [self._predict(video) for video in videos]

def run_embeddings_model(
    dataset,
    model_name,
    checkpoint,
    emb_field,
    embedding_types
    ):
    """
    Runs the Hiera embedding model on a FiftyOne dataset.

    Args:
        dataset (fo.Dataset): The FiftyOne dataset to process
        model_name (str): Name of the Hiera model to use
        checkpoint (str): Model checkpoint to use
        emb_field (str): Name of the field to store embeddings
        embedding_types (str): Type of embeddings to extract ('terminal' or 'hierarchical')
    """
    model = HieraVideoEmbeddingModel(
        model_name, 
        checkpoint, 
        embedding_types
        )

    dataset.apply_model(
        model, 
        label_field=emb_field
        )