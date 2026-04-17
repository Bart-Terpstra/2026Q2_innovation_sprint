# %%
import requests

import torch
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# %%
model_id = "IDEA-Research/grounding-dino-tiny"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

# %%
# image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(image_url, stream=True).raw)
# image_path = "../images/ship.jpg"
# image_path = "../images/rear_w_persons4_2025_CASE_OP_OS_4_STERE_BOIKY.JPG"
image_path = "../images/cropped_images/bridge_w_persons_2025_CASE_OP_OS_4_STERE_BOIKY.JPG"
image = Image.open(image_path)
# image = image.resize((768, 768))

# %%
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)
image = image.filter(ImageFilter.SHARPEN)

# %%
# Check for cats and remote controls
# text_labels = [["a remote control"]]
# text_labels = [["cannon", "gun turret", "naval gun"]]
# text_labels = [["a white box"]]
# text_labels = [["person", "people"]]
text_labels = [["antenna", "aerial", "thin and long"]]

# %%
inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# %%
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.15,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

result = results[0]
for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
    box = [round(x, 2) for x in box.tolist()]
    print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")

# %%
# Visualize the detected bounding boxes on top of the image.
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    x0, y0, x1, y1 = box.tolist()
    width = x1 - x0
    height = y1 - y0

    rect = patches.Rectangle(
        (x0, y0),
        width,
        height,
        linewidth=2,
        edgecolor="lime",
        facecolor="none"
    )
    ax.add_patch(rect)
    ax.text(
        x0,
        max(y0 - 5, 0),
        f"{label}: {score:.2f}",
        color="white",
        fontsize=10,
        bbox={"facecolor": "lime", "alpha": 0.7, "pad": 2}
    )

ax.axis("off")
plt.tight_layout()
plt.show()

# %%
