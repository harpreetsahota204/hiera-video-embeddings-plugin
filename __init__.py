import os

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

from .utils import HIERA_MODELS
from .embeddings import run_embeddings_model

def _handle_calling(
        uri, 
        sample_collection, 
        model_name,
        checkpoint,
        emb_field,
        embedding_types, 
        normalize,
        delegate=False
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_name=model_name,
        checkpoint=checkpoint,
        emb_field=emb_field,
        embedding_types=embedding_types,
        normalize=normalize,
        delegate=delegate
        )
    return foo.execute_operator(uri, ctx, params=params)

class HieraVideoEmbeddings(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            # The operator's URI: f"{plugin_name}/{name}"
            name="compute_hiera_video_embeddings",  # required

            # The display name of the operator
            label="Hiera Video Embeddings",  # required

            # A description for the operator
            description="Compute embeddings for video using a Hiera Model",

            icon="/assets/video-frame-2-svgrepo-com.svg",
            )

    def resolve_input(self, ctx):
        """Implement this method to collect user inputs as parameters
        that are stored in `ctx.params`.

        Returns:
            a `types.Property` defining the form's components
        """
        inputs = types.Object()

        model_dropdown = types.Dropdown(label="Choose the Hiera embedding model you want to use:")

        for arch_key, arch_value in HIERA_MODELS.items():
            model_dropdown.add_choice(arch_value, label=arch_key)

        inputs.enum(
            "model_name",
            values=model_dropdown.values(),
            label="Embedding Model",
            description="Select from one of the supported models. Note: The model weights will be downloaded from Torch Hub.",
            view=model_dropdown,
            required=True
        )

        embedding_types = types.RadioGroup(label="Which embedding approach do you want to use?",)

        embedding_types.add_choice(
            "terminal", 
            label="Terminal Embedding",
            description="A compact semantic representation extracted from the final layer of the model"
            )
        
        embedding_types.add_choice(
            "hierarchical", 
            label="Hierarchical Feature Embedding",
            description="A multi-scale representation that preserves information across different levels of abstraction"
            )
        
        inputs.enum(
            "embedding_types",
            values=embedding_types.values(),
            view=embedding_types,
            required=True
        )

        checkpoint = types.RadioGroup(label="Which checkpoint do you want to use?")

        checkpoint.add_choice(
            "mae_k400", 
            label="Pretrained",
            description="Model pretrained via MAE on Kinetics 400"
            )
        
        checkpoint.add_choice(
            "mae_k400_ft_k400", 
            label="Finetuned",
            description="Model finetuned on Kinetics Classes"
            )
        
        inputs.enum(
            "checkpoint",
            values=checkpoint.values(),
            view=checkpoint,
            required=True
        )

        inputs.bool(
            "normalize",
            default=False,
            required=True,
            label="Normalize embeddings?",
            description=(
                "Depending on your use case you may want to normalize embeddings."
                " Only applies when using `terminal` embeddings."),
            view=types.CheckboxView(),
        )

        inputs.str(
            "emb_field",            
            required=True,
            description="Name of the field to store the embeddings in."
            )
        
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        """Implement this method if you want to programmatically *force*
        this operation to be delegated or executed immediately.

        Returns:
            whether the operation should be delegated (True), run
            immediately (False), or None to defer to
            `resolve_execution_options()` to specify the available options
        """
        return ctx.params.get("delegate", False)


    def execute(self, ctx):
        """Executes the actual operation based on the hydrated `ctx`.
        All operators must implement this method.

        This method can optionally be implemented as `async`.

        Returns:
            an optional dict of results values
        """
        view = ctx.target_view()
        model_name = ctx.params.get("model_name")
        checkpoint = ctx.params.get("checkpoint")
        emb_field = ctx.params.get("emb_field")
        embedding_types = ctx.params.get("embedding_types")
        normalize = ctx.params.get("normalize")
        
        run_embeddings_model(
            dataset=view,
            model_name=model_name,
            checkpoint=checkpoint,
            emb_field=emb_field,
            embedding_types=embedding_types,
            normalize=normalize
            )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            model_name, 
            checkpoint,
            emb_field,
            embedding_types,
            normalize,
            delegate
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_name,
            checkpoint,
            emb_field,
            embedding_types,
            normalize,
            delegate
            )

def register(p):
    """Always implement this method and register() each operator that your
    plugin defines.
    """
    p.register(HieraVideoEmbeddings)