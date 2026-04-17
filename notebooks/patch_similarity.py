# %%
from IPython import get_ipython

get_ipython().run_line_magic("matplotlib", "ipympl")
get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import torch
from transformers import AutoModel
from huggingface_hub import login
try:
    import ipywidgets as Widget
except Exception:
    Widget = None  # slider is optional

# Support src-layout imports regardless of notebook/script launch directory.
def add_src_to_syspath() -> str | None:
    """Find and prepend the nearest repo src directory to sys.path."""
    search_roots = [Path.cwd().resolve()]
    if "__file__" in globals():
        search_roots.insert(0, Path(__file__).resolve().parent)

    for root in search_roots:
        for candidate in (root, *root.parents):
            src_dir = candidate / "src"
            if src_dir.exists():
                src_dir_str = str(src_dir)
                if src_dir_str not in sys.path:
                    sys.path.insert(0, src_dir_str)
                return src_dir_str
    return None

add_src_to_syspath()

from similarity_search.image.processing import (
    load_image,
)
from similarity_search.patch.features import (
    build_state,
)
from similarity_search.visualization.patch_similarity_viewer import (
    PatchSimilarityViewer,
)

# loads .env from project root, where HF_TOKEN should be set to a HuggingFace 
# token with access to the model (can be read-only)
load_dotenv()
token = os.getenv("HF_TOKEN")
# Login to HuggingFace Hub (for model loading)
login(token)

# %%
# setting device to first of these: [GPU, MPS (MacOS), CPU]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
print(f"Using Torch device '{device}'")

# %%
# Define the images to compare and their labels (for display)
path_left = "../images/flags/1.JPG"
label_left = "iteration 1 image 1"
path_right ="../images/flags/2.JPG"
label_right = "iteration 1 image 2"
image_left = load_image(path_left)
image_right = load_image(path_right)


# %%
# Load the pretrained model (DINOv3 ViT-S/16 in this case, but should work with any ViT-based model with patch embeddings in the last hidden state)
pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"  # Load locally
model = AutoModel.from_pretrained(pretrained_model_name).to(device)
model.eval()

# Set the patch size
patch_size_override = None  # set to 16 to force; None = read from model if available 
ps = patch_size_override if patch_size_override is not None else getattr(getattr(model, "config", object()), "patch_size", 16)

# %%
# Patch similarity visualization
if image_right is not None:
    states = [
        build_state(image_left, model, ps, device),
        build_state(image_right, model, ps, device),
    ]
    labels = [label_left, label_right]
else:
    states = [build_state(image_left, model, ps, device)]
    labels = [label_left]

# Configure visualization options
show_grid = False
show_overlay = True
annotate_indices = False
overlay_alpha = 0.55

viewer = PatchSimilarityViewer(
    states=states,
    labels=labels,
    show_grid=show_grid,
    show_overlay=show_overlay,
    overlay_alpha=overlay_alpha,
    annotate_indices=annotate_indices,
)
viewer.show()


# %%

