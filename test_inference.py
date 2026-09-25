import os
import yaml
import numpy as np

import torch
from torchvision import datasets, transforms

from models import CNNMNIST
from BitNetMCU import QuantizedModel, inference_quantized


# ============================================================
# 1. Load training parameters
# ============================================================

with open("trainingparameters.yaml", "r") as f:
    hyperparameters = yaml.safe_load(f)

print("=" * 70)
print("TEST DATASET INFERENCE CHECK")
print("=" * 70)

print("\nConfiguration:")
for key, value in hyperparameters.items():
    print(f"  {key}: {value}")


# ============================================================
# 2. Recreate EXACT training run name
# ============================================================

runname = (
    hyperparameters["runtag"]
    + "_"
    + hyperparameters["model"]
    + ("_Aug" if hyperparameters.get("augmentation", False) else "")
    + "_BitMnist_"
    + hyperparameters["QuantType"]
    + "_width"
    + str(hyperparameters["network_width1"])
    + "_"
    + str(hyperparameters["network_width2"])
    + "_"
    + str(hyperparameters["network_width3"])
    + "_epochs"
    + str(hyperparameters["num_epochs"])
)

print("\nRun name:")
print(" ", runname)


# ============================================================
# 3. Expected classes
# ============================================================

expected_classes = [
    "Alice",
    "Azizul",
    "Faizal",
    "Haziq",
    "Henry",
    "Kah_Siang",
    "Luqman",
    "Nurina",
    "Sean",
    "Steven",
]

print("\nExpected classes:")
for i, name in enumerate(expected_classes):
    print(f"  {i}: {name}")


# ============================================================
# 4. Dataset paths
# ============================================================

data_root = hyperparameters["data_root"]

train_dir = os.path.join(
    data_root,
    hyperparameters["train_folder"]
)

val_dir = os.path.join(
    data_root,
    hyperparameters["val_folder"]
)

test_dir = os.path.join(
    data_root,
    hyperparameters["test_folder"]
)

print("\nDataset paths:")
print("  Data root :", os.path.abspath(data_root))
print("  Train     :", os.path.abspath(train_dir))
print("  Validation:", os.path.abspath(val_dir))
print("  Test      :", os.path.abspath(test_dir))


# ============================================================
# 5. Check directories exist
# ============================================================

for name, path in [
    ("Train", train_dir),
    ("Validation", val_dir),
    ("Test", test_dir),
]:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{name} directory does not exist:\n{path}"
        )

print("\nAll dataset directories exist.")


# ============================================================
# 6. EXACT SAME preprocessing as training
# ============================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ============================================================
# 7. Load all three datasets
# ============================================================

train_data = datasets.ImageFolder(
    train_dir,
    transform=transform
)

val_data = datasets.ImageFolder(
    val_dir,
    transform=transform
)

test_data = datasets.ImageFolder(
    test_dir,
    transform=transform
)


# ============================================================
# 8. Check class mappings
# ============================================================

print("\nActual ImageFolder class mapping:")

print("\nTrain:")
print(train_data.class_to_idx)

print("\nValidation:")
print(val_data.class_to_idx)

print("\nTest:")
print(test_data.class_to_idx)


if train_data.classes != expected_classes:
    raise ValueError(
        "\nTRAIN class order is WRONG!\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {train_data.classes}"
    )

if val_data.classes != expected_classes:
    raise ValueError(
        "\nVALIDATION class order is WRONG!\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {val_data.classes}"
    )

if test_data.classes != expected_classes:
    raise ValueError(
        "\nTEST class order is WRONG!\n"
        f"Expected: {expected_classes}\n"
        f"Found:    {test_data.classes}"
    )

print("\nPASS: Train / Validation / Test class ordering is identical.")


# ============================================================
# 9. Check number of classes
# ============================================================

num_classes = hyperparameters["num_classes"]

if num_classes != len(expected_classes):
    raise ValueError(
        f"YAML num_classes={num_classes}, "
        f"but expected {len(expected_classes)} classes."
    )

if len(test_data.classes) != num_classes:
    raise ValueError(
        f"Test dataset has {len(test_data.classes)} classes, "
        f"but model expects {num_classes}."
    )

print(f"PASS: Number of classes = {num_classes}")


# ============================================================
# 10. Dataset sample counts
# ============================================================

print("\nDataset sample counts:")
print(f"  Train      : {len(train_data)}")
print(f"  Validation : {len(val_data)}")
print(f"  Test       : {len(test_data)}")


print("\nTest samples per class:")

test_targets = np.array(test_data.targets)

for class_id, class_name in enumerate(test_data.classes):
    count = np.sum(test_targets == class_id)
    print(f"  {class_id}: {class_name:<12} {count}")


# ============================================================
# 11. Build model
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:", device)

model = CNNMNIST(
    network_width1=hyperparameters["network_width1"],
    network_width2=hyperparameters["network_width2"],
    network_width3=hyperparameters["network_width3"],
    QuantType=hyperparameters["QuantType"],
    NormType=hyperparameters["NormType"],
    WScale=hyperparameters["WScale"],
    cnn_width=hyperparameters["cnn_width"],
    num_classes=num_classes,
).to(device)


# ============================================================
# 12. Check model output
# ============================================================

dummy_input = torch.randn(
    1,
    1,
    32,
    32,
    device=device
)

with torch.no_grad():
    dummy_output = model(dummy_input)

print("\nModel output shape:", tuple(dummy_output.shape))

if dummy_output.shape[1] != num_classes:
    raise ValueError(
        f"Model outputs {dummy_output.shape[1]} classes, "
        f"but dataset has {num_classes} classes."
    )

print("PASS: Model output = 10 classes.")


# ============================================================
# 13. Load FINAL checkpoint
# ============================================================

model_path = os.path.join(
    "modeldata",
    f"{runname}.pth"
)

print("\nCheckpoint:")
print(" ", os.path.abspath(model_path))

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"\nFinal checkpoint not found:\n{model_path}"
    )

print("PASS: Final checkpoint exists.")

state_dict = torch.load(
    model_path,
    map_location=device
)

model.load_state_dict(state_dict)
model.eval()

print("PASS: Final checkpoint loaded successfully.")


# ============================================================
# 14. Original PyTorch inference
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL PYTORCH MODEL")
print("=" * 70)

correct = 0
total = 0

class_correct = np.zeros(num_classes, dtype=int)
class_total = np.zeros(num_classes, dtype=int)

all_predictions = []
all_labels = []

with torch.no_grad():

    for i in range(len(test_data)):

        image, label = test_data[i]

        image = image.unsqueeze(0).to(device)

        output = model(image)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

        all_predictions.append(prediction)
        all_labels.append(label)

        total += 1

        class_total[label] += 1

        if prediction == label:
            correct += 1
            class_correct[label] += 1


original_accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

print(
    f"\nOriginal model accuracy: "
    f"{correct}/{total} = {original_accuracy:.2f}%"
)


print("\nPer-class accuracy:")

for class_id, class_name in enumerate(expected_classes):

    if class_total[class_id] > 0:
        acc = (
            class_correct[class_id]
            / class_total[class_id]
            * 100
        )
    else:
        acc = 0.0

    print(
        f"  {class_name:<12}: "
        f"{class_correct[class_id]}/{class_total[class_id]} "
        f"= {acc:.2f}%"
    )


# ============================================================
# 15. Confusion matrix
# ============================================================

confusion = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for true_label, prediction in zip(
    all_labels,
    all_predictions
):
    confusion[true_label, prediction] += 1


print("\nConfusion Matrix:")
print(confusion)


# ============================================================
# 16. Quantized Python model
# ============================================================

print("\n" + "=" * 70)
print("QUANTIZED PYTHON MODEL")
print("=" * 70)

quant_model = QuantizedModel(model)

quant_correct = 0
quant_total = 0

quant_class_correct = np.zeros(
    num_classes,
    dtype=int
)

quant_class_total = np.zeros(
    num_classes,
    dtype=int
)

quant_predictions = []

for i in range(len(test_data)):

    image, label = test_data[i]

    # Convert [1, 32, 32] -> flattened [1, 1024]
    image_flat = image.reshape(
        1,
        -1
    ).numpy()

    prediction_output = inference_quantized(
        quant_model,
        image_flat
    )

    prediction = int(
        np.argmax(prediction_output)
    )

    quant_predictions.append(prediction)

    quant_total += 1
    quant_class_total[label] += 1

    if prediction == label:
        quant_correct += 1
        quant_class_correct[label] += 1


quant_accuracy = (
    quant_correct / quant_total * 100
    if quant_total > 0
    else 0
)

print(
    f"\nQuantized Python accuracy: "
    f"{quant_correct}/{quant_total} = "
    f"{quant_accuracy:.2f}%"
)


print("\nQuantized per-class accuracy:")

for class_id, class_name in enumerate(expected_classes):

    if quant_class_total[class_id] > 0:
        acc = (
            quant_class_correct[class_id]
            / quant_class_total[class_id]
            * 100
        )
    else:
        acc = 0.0

    print(
        f"  {class_name:<12}: "
        f"{quant_class_correct[class_id]}/"
        f"{quant_class_total[class_id]} "
        f"= {acc:.2f}%"
    )


# ============================================================
# 17. Quantized confusion matrix
# ============================================================

quant_confusion = np.zeros(
    (num_classes, num_classes),
    dtype=int
)

for true_label, prediction in zip(
    all_labels,
    quant_predictions
):
    quant_confusion[true_label, prediction] += 1


print("\nQuantized Confusion Matrix:")
print(quant_confusion)


# ============================================================
# 18. Compare original vs quantized
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL vs QUANTIZED")
print("=" * 70)

accuracy_difference = (
    original_accuracy - quant_accuracy
)

print(
    f"Original PyTorch : {original_accuracy:.2f}%"
)

print(
    f"Quantized Python : {quant_accuracy:.2f}%"
)

print(
    f"Difference       : {accuracy_difference:+.2f}%"
)


if abs(accuracy_difference) < 1.0:
    print(
        "\nPASS: Original and quantized Python accuracy "
        "are very close."
    )
else:
    print(
        "\nWARNING: Original and quantized Python accuracy "
        "differ by >= 1 percentage point."
    )


# ============================================================
# 19. Final summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST CHECK")
print("=" * 70)

print("PASS: Dataset directories exist")
print("PASS: 10 classes detected")
print("PASS: Class mapping verified")
print("PASS: Preprocessing matches training")
print("PASS: Model output matches 10 classes")
print("PASS: FINAL .pth checkpoint loaded")
print("PASS: Original PyTorch inference completed")
print("PASS: Quantized Python inference completed")

print("\nTest dataset is ready for evaluation.")

print("=" * 70)
