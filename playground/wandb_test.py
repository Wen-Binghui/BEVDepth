import wandb
import random
from PIL import Image as PILImage
import numpy as np
epochs = 100
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    notes="My first experiment",
    tags=["baseline", "paper1"],
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "batch_size": 128,
    "epochs": epochs
    }
)

# simulate training

offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch {epoch}")
    # log metrics to wandb
    
    pixels = np.random.randint(low=0, high=256, size=(100, 100, 3), dtype=np.uint8)
    pil_image = PILImage.fromarray(pixels, mode="RGB")
    image = wandb.Image(pil_image, caption=f"random field {epoch}")

    wandb.log({"acc": acc, "loss": loss, "image": image})
# [optional] finish the wandb run, necessary in notebooks
wandb.finish()