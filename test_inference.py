%%writefile test_inference.py

import os
import argparse
import importlib

import yaml
import numpy as np
import torch

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from BitNetMCU import QuantizedModel


def create_run_name(hyperparameters):
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

    hyperparameters["runname"] = runname
    return runname


def load_model(model_name, params):

    module = importlib.import_module("models")
    model_class = getattr(module, model_name)

    kwargs = dict(
        network_width1=params["network_width1"],
        network_width2=params["network_width2"],
        network_width3=params["network_width3"],
        QuantType=params["QuantType"],
        NormType=params["NormType"],
        WScale=params["WScale"],
    )

    if "cnn_width" in params:
        kwargs["cnn_width"] = params["cnn_width"]

    if "num_classes" in params:
        kwargs["num_classes"] = params["num_classes"]

    return model_class(**kwargs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FACE 32x32 test inference"
    )

    parser.add_argument(
        "--params",
        type=str,
        default="trainingparameters.yaml"
    )

    args = parser.parse_args()

    with open(args.params) as f:
        hyperparameters = yaml.safe_load(f)

    hyperparameters.setdefault("augmentation", False)
    hyperparameters.setdefault("network_width3", 0)

    runname = create_run_name(hyperparameters)

    print("Run name:", runname)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # FACE dataset
    # --------------------------------------------------------

    dataset_name = hyperparameters.get(
        "dataset", "FACE"
    ).upper()

    if dataset_name != "FACE":
        raise ValueError(
            f"This script is for FACE only. "
            f"Current dataset: {dataset_name}"
        )

    data_root = hyperparameters["data_root"]
    test_folder = hyperparameters.get("test_folder", "test")

    test_dir = os.path.join(
        data_root,
        test_folder
    )

    print("FACE TEST folder:", test_dir)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(
            f"Test folder not found:\n{test_dir}"
        )

    # Same preprocessing as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_data = ImageFolder(
        root=test_dir,
        transform=transform
    )

    num_classes = len(test_data.classes)

    print("Test samples:", len(test_data))
    print("Number of classes:", num_classes)
    print("Class mapping:", test_data.class_to_idx)

    if num_classes != hyperparameters["num_classes"]:
        raise ValueError(
            f"Expected {hyperparameters['num_classes']} classes, "
            f"but found {num_classes} classes."
        )

    test_loader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )

    hyperparameters["num_classes"] = num_classes

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = load_model(
        hyperparameters["model"],
        hyperparameters
    ).to(device)

    # --------------------------------------------------------
    # Check input/output
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1, 1, 32, 32
    ).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print("Model input shape:", dummy_input.shape)
    print("Model output shape:", dummy_output.shape)
    print("Expected classes:", num_classes)

    if dummy_output.shape[1] != num_classes:
        raise ValueError(
            f"Model output mismatch. "
            f"Model outputs {dummy_output.shape[1]} classes, "
            f"but dataset has {num_classes} classes."
        )
        
    # --------------------------------------------------------
    # Load FINAL model
    # --------------------------------------------------------

    model_path = f"modeldata/{runname}.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find final model checkpoint:\n{model_path}"
    )

    model.load_state_dict(
    torch.load(
            model_path,
            map_location=torch.device("cpu")
        )
    )

    print("Loaded checkpoint:", model_path)

    model = model.to(device)
    model.eval()
    
    # --------------------------------------------------------
    # Original model inference
    # --------------------------------------------------------

    print()
    print("========================================")
    print("Original model inference")
    print("========================================")

    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

            all_predictions.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    test_accuracy = correct / total * 100

    print(f"Test samples: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {total - correct}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print()
    print("========================================")
    print("Per-class results")
    print("========================================")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    for class_name, class_index in test_data.class_to_idx.items():

        mask = all_labels == class_index

        class_total = mask.sum()

        class_correct = (
            all_predictions[mask] == class_index
        ).sum()

        class_accuracy = (
            class_correct / class_total * 100
            if class_total > 0
            else 0
        )

        print(
            f"{class_name:15s} : "
            f"{class_correct}/{class_total} "
            f"({class_accuracy:.2f}%)"
        )

    # --------------------------------------------------------
    # Quantized Python model
    # --------------------------------------------------------

    print()
    print("========================================")
    print("Quantized Python model")
    print("========================================")

    quantized_model = QuantizedModel(model)

    total_bits = quantized_model.totalbits()

    print(f"Total bits: {total_bits}")
    print(
        f"Model size: "
        f"{total_bits / 8 / 1024:.4f} KB"
    )

    test_loader_quantized = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False
    )

    counter = 0
    correct_quantized = 0

    for input_data, labels in test_loader_quantized:

        input_numpy = (
            input_data
            .view(input_data.size(0), -1)
            .cpu()
            .numpy()
        )

        labels_numpy = labels.cpu().numpy()

        result_quantized = (
            quantized_model.inference_quantized(
                input_numpy
            )
        )

        prediction_quantized = np.argmax(
            result_quantized,
            axis=1
        )

        if (
            prediction_quantized[0]
            == labels_numpy[0]
        ):
            correct_quantized += 1

        counter += 1

    quantized_accuracy = (
        correct_quantized / counter * 100
    )

    print()
    print("========================================")
    print("Quantized Python results")
    print("========================================")

    print(f"Test samples: {counter}")
    print(f"Correct: {correct_quantized}")
    print(f"Wrong: {counter - correct_quantized}")
    print(
        f"Quantized Python Accuracy: "
        f"{quantized_accuracy:.2f}%"
    )

    print()
    print("Test inference completed.")
