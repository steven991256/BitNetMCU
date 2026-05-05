import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

from pathlib import Path
from ctypes import CDLL, c_uint32, c_int8, POINTER

import argparse
import yaml
import importlib

from BitNetMCU import QuantizedModel


# ------------------------------------------------------------
# Run name
# ------------------------------------------------------------

def create_run_name(hyperparameters):
    dataset_name = hyperparameters.get("dataset", "MNIST")

    runname = (
        hyperparameters["runtag"]
        + "_"
        + hyperparameters["model"]
        + ("_Aug" if hyperparameters["augmentation"] else "")
        + "_"
        + dataset_name
        + "_"
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


# ------------------------------------------------------------
# Model loader
# ------------------------------------------------------------

def load_model(model_name, params):
    try:
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

        if "input_size" in params:
            kwargs["input_size"] = params["input_size"]

        return model_class(**kwargs)

    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py")


# ------------------------------------------------------------
# Signal CSV dataset
# ------------------------------------------------------------

class SignalDataset(Dataset):
    """
    CSV signal dataset.

    Expected structure:

        signal_dataset/split/test/
            idle/
            left/
            right/

    Each CSV file:

        ax,ay,az
        10,20,16384
        ...
    """

    def __init__(
        self,
        root_dir,
        class_names=None,
        samples_per_window=64,
        channels=3,
        normalize_scale=16384.0,
    ):
        self.root_dir = Path(root_dir)
        self.samples_per_window = int(samples_per_window)
        self.channels = int(channels)
        self.normalize_scale = float(normalize_scale)

        if class_names is None:
            class_names = ["idle", "left", "right"]

        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(f"Missing signal class folder: {class_dir}")

            for csv_file in sorted(class_dir.glob("*.csv")):
                self.samples.append((csv_file, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No CSV files found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csv_file, label = self.samples[idx]

        df = pd.read_csv(csv_file)

        expected_columns = ["ax", "ay", "az"]
        for col in expected_columns:
            if col not in df.columns:
                raise ValueError(f"{csv_file} is missing column: {col}")

        data = df[expected_columns].values.astype(np.float32)

        if data.shape[0] > self.samples_per_window:
            data = data[:self.samples_per_window]
        elif data.shape[0] < self.samples_per_window:
            pad = np.zeros(
                (self.samples_per_window - data.shape[0], self.channels),
                dtype=np.float32,
            )
            data = np.vstack([data, pad])

        # Normalize MPU6050-style values.
        data = data / self.normalize_scale

        # 64 x 3 -> 192
        data = data.flatten()

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


# ------------------------------------------------------------
# Optional C test-data export
# ------------------------------------------------------------

def export_test_data_to_c(test_loader, filename, input_size, num=8):
    with open(filename, "w") as f:
        for i, (input_data, labels) in enumerate(test_loader):
            if i >= num:
                break

            input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
            labels = labels.cpu().numpy()

            scale = 127.0 / np.maximum(
                np.abs(input_data).max(axis=-1, keepdims=True),
                1e-5,
            )

            scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.int8)

            f.write(f"int8_t input_data_{i}[{input_size}] = {{\n")

            flattened_data = scaled_data.flatten()

            for k in range(0, len(flattened_data), 16):
                values = flattened_data[k:k + 16]
                f.write(", ".join(str(int(v)) for v in values) + ",\n")

            f.write("};\n")
            f.write(f"uint8_t label_{i} = {int(labels[0])};\n\n")


# ------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------

def build_test_dataset(hyperparameters):
    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    if dataset_name == "SIGNAL":
        signal_classes = hyperparameters.get(
            "signal_classes",
            ["idle", "left", "right"],
        )

        signal_test_dir = hyperparameters.get(
            "signal_test_dir",
            "signal_dataset/split/test",
        )

        samples_per_window = hyperparameters.get("samples_per_window", 64)
        signal_channels = hyperparameters.get("signal_channels", 3)
        normalize_scale = hyperparameters.get("signal_normalize_scale", 16384.0)

        num_classes = len(signal_classes)
        input_size = samples_per_window * signal_channels

        test_data = SignalDataset(
            root_dir=signal_test_dir,
            class_names=signal_classes,
            samples_per_window=samples_per_window,
            channels=signal_channels,
            normalize_scale=normalize_scale,
        )

        hyperparameters["num_classes"] = num_classes
        hyperparameters["input_size"] = input_size

        print("Loaded SIGNAL test dataset")
        print("Classes:", signal_classes)
        print("Test samples:", len(test_data))
        print("Input size:", input_size)

        return test_data

    if dataset_name == "MNIST":
        mean, std = (0.1307,), (0.3081,)
        dataset_cls = datasets.MNIST

    elif dataset_name == "FASHION":
        mean, std = (0.2860,), (0.3530,)
        dataset_cls = datasets.FashionMNIST

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    transform = transforms.Compose(
        [
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_data = dataset_cls(
        root="data",
        train=False,
        transform=transform,
        download=True,
    )

    hyperparameters["num_classes"] = 10
    hyperparameters["input_size"] = 256

    print(f"Loaded {dataset_name} test dataset")
    print("Test samples:", len(test_data))
    print("Input size:", hyperparameters["input_size"])

    return test_data


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BitNetMCU inference test script")

    parser.add_argument(
        "--params",
        type=str,
        help="Name of the parameter file",
        default="trainingparameters.yaml",
    )

    parser.add_argument(
        "--dll",
        type=str,
        help="Path to compiled C inference DLL/shared library",
        default="./Bitnet_inf.dll",
    )

    parser.add_argument(
        "--skip-c",
        action="store_true",
        help="Only test PyTorch and Python quantized inference, skip C DLL test",
    )

    parser.add_argument(
        "--export-c-test-data",
        action="store_true",
        help="Export several quantized test vectors to a C header",
    )

    args = parser.parse_args()

    paramname = args.params if args.params else "trainingparameters.yaml"

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    runname = create_run_name(hyperparameters)
    print("Run name:", runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_data = build_test_dataset(hyperparameters)

    test_loader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
    )

    test_loader_single = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
    )

    input_size = hyperparameters["input_size"]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(hyperparameters["model"], hyperparameters).to(device)

    print("Loading model...")

    model_path = f"modeldata/{runname}.pth"

    try:
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu"),
            )
        )
    except FileNotFoundError:
        print(f"The file '{model_path}' does not exist.")
        exit(1)

    model.eval()

    # --------------------------------------------------------
    # Original PyTorch model inference
    # --------------------------------------------------------

    print("Inference using the original model...")

    correct = 0
    total = 0

    with torch.no_grad():
        for input_data, labels in test_loader:
            input_data = input_data.to(device)
            labels = labels.to(device)

            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testaccuracy = correct / total * 100.0

    print(f"Accuracy/Test of trained model: {testaccuracy:.2f} %")

    # --------------------------------------------------------
    # Quantized model
    # --------------------------------------------------------

    print("Quantizing model...")

    quantized_model = QuantizedModel(model)

    print(
        f"Total number of bits: {quantized_model.totalbits()} "
        f"({quantized_model.totalbits() / 8 / 1024:.4f} kbytes)"
    )

    if args.export_c_test_data:
        if dataset_name == "SIGNAL":
            filename = "BitNetMCU_SIGNAL_test_data.h"
        else:
            filename = "BitNetMCU_MNIST_test_data.h"

        export_test_data_to_c(
            test_loader_single,
            filename,
            input_size=input_size,
            num=10,
        )

        print(f"Exported C test data to {filename}")

    # --------------------------------------------------------
    # Verify Python quantized inference
    # --------------------------------------------------------

    print("Verifying inference of quantized model in Python")

    counter = 0
    correct_py = 0

    py_predictions = []
    true_labels = []

    for input_data, labels in test_loader_single:
        input_np = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        result_py = quantized_model.inference_quantized(input_np)
        predict_py = np.argmax(result_py, axis=1)

        if predict_py[0] == labels_np[0]:
            correct_py += 1

        py_predictions.append(int(predict_py[0]))
        true_labels.append(int(labels_np[0]))

        counter += 1

    print("Size of test data:", counter)
    print(f"Mispredictions Python: {counter - correct_py}")
    print(f"Overall accuracy Python: {correct_py / counter * 100:.2f} %")

    # --------------------------------------------------------
    # Optional C DLL inference
    # --------------------------------------------------------

    if args.skip_c:
        print("Skipping C DLL inference because --skip-c was used.")
        exit(0)

    print("Verifying inference of quantized model in C")

    try:
        lib = CDLL(args.dll)
    except OSError as e:
        print(f"Could not load C inference library: {args.dll}")
        print(e)
        print("Use --skip-c if you only want Python quantized verification.")
        exit(1)

    lib.Inference.argtypes = [POINTER(c_int8)]
    lib.Inference.restype = c_uint32

    counter = 0
    correct_c = 0
    correct_py = 0
    mismatch = 0

    for input_data, labels in test_loader_single:
        input_np = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        scale = 127.0 / np.maximum(
            np.abs(input_np).max(axis=-1, keepdims=True),
            1e-5,
        )

        scaled_data = np.round(input_np * scale).clip(-128, 127).astype(np.int8)

        input_data_pointer = (c_int8 * len(scaled_data.flatten()))(
            *scaled_data.flatten()
        )

        result_c = lib.Inference(input_data_pointer)

        result_py = quantized_model.inference_quantized(input_np)
        predict_py = np.argmax(result_py, axis=1)

        true_label = int(labels_np[0])
        pred_py = int(predict_py[0])
        pred_c = int(result_c)

        if pred_c == true_label:
            correct_c += 1

        if pred_py == true_label:
            correct_py += 1

        if pred_c != pred_py:
            print(
                f"{counter:5} Mismatch between inference engines found. "
                f"Prediction C: {pred_c} "
                f"Prediction Python: {pred_py} "
                f"True: {true_label}"
            )
            mismatch += 1

        counter += 1

    print("Size of test data:", counter)
    print(f"Mispredictions C: {counter - correct_c} Py: {counter - correct_py}")
    print(f"Overall accuracy C: {correct_c / counter * 100:.2f} %")
    print(f"Overall accuracy Python: {correct_py / counter * 100:.2f} %")
    print(f"Mismatches between engines: {mismatch} ({mismatch / counter * 100:.2f}%)")
