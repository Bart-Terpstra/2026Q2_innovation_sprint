# %%
import geoai
import torch
import os
print(torch.cuda.is_available()) # indicates whether a GPU is available for PyTorch

# ## Download sample data
# %%
url = (
    "https://huggingface.co/datasets/giswqs/geospatial/resolve/main/naip_rgb_train.tif"
)
local_path = os.path.basename(url)
raster_path = local_path if os.path.exists(local_path) else geoai.download_file(url)

# ## Initiliaze the DINOv3 processor
# %%
processor = geoai.DINOv3GeoProcessor(
    model_name="dinov3_vitl16",
)

# ## Extract features
# %%
image = "naip_rgb_train.tif"
features, h_patches, w_patches = processor.extract_features(image)

# ## Find similar patches
# %%
m = geoai.LeafMap()
m.add_dinov3_gui(image, processor, features)
m