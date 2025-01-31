## Facebook Hiera Video Embeddings Plugins

### Plugin Overview

This plugin allows you to compute embeddings on videos using [Facebook's Hiera](https://github.com/facebookresearch/hiera) models on your FiftyOne datasets.

**Note*:* This plugin only computing supports video embeddings. 

#### Supported Models

This plugin supports all currently released versions and checkpoints of the [Hiera Video Models collection](https://github.com/facebookresearch/hiera):

    - `hiera_base_16x224`
    - `hiera_base_plus_16x224`
    - `hiera_large_16x224`
    - `hiera_huge_16x224`

## Installation

If you haven't already, install FiftyOne:

```shell
pip install -U fiftyone 
```

Then, install the plugin:

```shell
fiftyone plugins download https://github.com/harpreetsahota204/hiera-video-embeddings-plugin
```

To install requirements:

```shell
fiftyone plugins requirements @harpreetsahota/hiera_video_embeddings --install
```

In order to use the plugin you need to have the following enviornment variable set:

```python
import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'
```

### Embedding Types

The plugin supports two types of embeddings:

- **Terminal Embedding (`terminal`)**: A 768-dimensional embedding vector derived from the final layer of the model. This represents the global semantic context of the video sequence. Can optionally be normalized.
  
- **Hierarchical Embedding (`hierarchical`)**: A 1440-dimensional embedding vector that concatenates features across all intermediate outputs (96+192+384+768 = 1440 dimensions). This captures multi-scale representations of the video content. These embeddings cannot be normalized.

### Technical Details

The model processes videos with the following specifications:
- Input shape: (T, H, W, C) where:
  - T: number of frames
  - H: frame height
  - W: frame width
  - C: number of channels (3 for RGB)
- Output shape:
  - Terminal embeddings: (768,)
  - Hierarchical embeddings: (1440,)

The model automatically handles video preprocessing and can utilize GPU acceleration when available.

#### Video Processing Pipeline

The plugin automatically handles video preprocessing through the following steps:

1. **Frame Loading**: Video frames are loaded and converted to numpy arrays
2. **Tensor Conversion**: Frames are stacked and converted to PyTorch tensors
3. **Dimension Arrangement**: 
   - Input: `[T, H, W, C]` (Time, Height, Width, Channels)
   - Transformed to: `[1, C, T, H, W]` (Batch, Channels, Time, Height, Width)
4. **Resizing**: Videos are automatically resized to:
   - 16 frames (temporal dimension)
   - 224x224 pixels (spatial dimensions)
   - Uses trilinear interpolation for smooth resizing

#### Model Input/Output Specifications

**Input Format:**
- Shape: `(1, 3, 16, 224, 224)`
  - Batch size: 1
  - Channels: 3 (RGB)
  - Frames: 16
  - Height: 224
  - Width: 224

**Output Format:**
- Terminal embeddings: `(768,)`
- Hierarchical embeddings: `(1440,)` (concatenated features: 96+192+384+768)


## Usage in FiftyOne App

You can compute Hiera embeddings directly through the FiftyOne App:

1. Launch the FiftyOne App with your dataset
2. Open the "Operators Browser" by clicking on the Operator Browser icon or pressing the backtick (`) key
3. Search for "compute_hiera_embeddings"
4. Configure the following parameters:
   - **Model Name**: The Hiera model architecture to use
   - **Checkpoint**: Model checkpoint identifier
   - **Embedding Type**: Choose between:
     - `terminal` - Final layer embedding (768-dim)
     - `hierarchical` - Multi-scale embedding (1440-dim)
   - **Field Name**: Enter the name for the embeddings field
   - **Normalize**: (Optional and only apples to `terminal` embeddings) Whether to L2-normalize terminal embeddings 
5. After embeddings are computed you **must** call `dataset.reload()` to have them registered to your dataset.

The embeddings will be stored in the specified field name and can be used for similarity searches, visualization, or other downstream tasks. 

## Operators

### `compute_hiera_video_embeddings`

This operator computes video embeddings using a Hiera model.

## Operator usage via SDK

Once the plugin has been installed, you can instantiate the operator as follows:

```python
import fiftyone.operators as foo

embedding_operator = foo.get_operator("@harpreetsahota/hiera_video_embeddings/compute_hiera_video_embeddings")
```

You can then compute embeddings on your dataset by running the operator with your desired parameters:

```python
# Run the operator on your dataset
embedding_operator(
    dataset,
    model_name="<choose-any-supported-model>",
    checkpoint="<choose any checkpoint, one of mae_k400 OR mae_k400_ft_k400>",
    embedding_types="terminal", #or hierarchical
    emb_field="<name-of-field>",
    normalize=False, #defaults to False, only works with `terminal` embeddings
    delegate=True
    )
```

If you're running in a notebook, it's recommended to launch a [Delegated operation](https://docs.voxel51.com/plugins/using_plugins.html#delegated-operations) by running `fiftyone delegated launch` in terminal, then run as follows:

```python
await embedding_operator(
    dataset,
    model_name="<choose-any-supported-model>",
    checkpoint="<choose any checkpoint, one of mae_k400 OR mae_k400_ft_k400>",
    embedding_types="terminal", #or hierarchical
    emb_field="<name-of-field>",
    normalize=False, #defaults to False, only works with `terminal` embeddings
    delegate=True
    )
```

### ℹ️ **Important:** You must call `dataset.reload()` to have the embeddings registered to your Dataset.

# Citation

You can read the paper here.

```bibtex
@article{ryali2023hiera,
  title={Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles},
  author={Ryali, Chaitanya and Hu, Yuan-Ting and Bolya, Daniel and Wei, Chen and Fan, Haoqi and Huang, Po-Yao and Aggarwal, Vaibhav and Chowdhury, Arkabandhu and Poursaeed, Omid and Hoffman, Judy and Malik, Jitendra and Li, Yanghao and Feichtenhofer, Christoph},
  journal={ICML},
  year={2023}
}
```