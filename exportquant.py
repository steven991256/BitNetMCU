import os
import json
import pickle
import argparse
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from BitNetMCU import QuantizedModel, BitLinear, BitConv2d, Activation

try:
    import importlib
except Exception:
    importlib = None


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


def make_bitlinear(in_features, out_features, QuantType, NormType, WScale):
    try:
        return BitLinear(
            in_features,
            out_features,
            QuantType=QuantType,
            NormType=NormType,
            WScale=WScale,
        )
    except TypeError:
        try:
            return BitLinear(
                in_features,
                out_features,
                QuantType,
                NormType,
                WScale,
            )
        except TypeError:
            print("Warning: BitLinear failed. Falling back to nn.Linear.")
            return nn.Linear(in_features, out_features)


class GateDriverMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        network_width1=64,
        network_width2=32,
        network_width3=0,
        QuantType="4bitsym",
        NormType="RMS",
        WScale="PerTensor",
        num_classes=4,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        layers = []

        layers.append(make_bitlinear(input_dim, network_width1, QuantType, NormType, WScale))
        layers.append(Activation())

        layers.append(make_bitlinear(network_width1, network_width2, QuantType, NormType, WScale))
        layers.append(Activation())

        if network_width3 and network_width3 > 0:
            layers.append(make_bitlinear(network_width2, network_width3, QuantType, NormType, WScale))
            layers.append(Activation())
            layers.append(make_bitlinear(network_width3, num_classes, QuantType, NormType, WScale))
        else:
            layers.append(make_bitlinear(network_width2, num_classes, QuantType, NormType, WScale))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model(model_name, params):
    if model_name == "GateDriverMLP":
        return GateDriverMLP(**params)

    if importlib is None:
        raise ValueError("Cannot import models.py")

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

        return model_class(**kwargs)

    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py or exportquant.py")


def tabular_to_cnnmnist_image(X):
    if X.shape[1] > 256:
        raise ValueError(f"Too many features for 16x16 CNN input: {X.shape[1]} > 256")

    X_pad = np.zeros((X.shape[0], 256), dtype=np.float32)
    X_pad[:, :X.shape[1]] = X
    return X_pad.reshape(-1, 1, 16, 16)


def load_gate_driver_test_excel(test_file, label_col="label", model_name="GateDriverMLP"):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test Excel file not found: {test_file}")

    feature_cols_path = "modeldata/gate_driver_feature_cols.json"
    label_mapping_path = "modeldata/gate_driver_label_mapping.json"
    scaler_path = "modeldata/gate_driver_scaler.pkl"

    if not os.path.exists(feature_cols_path):
        raise FileNotFoundError("Missing modeldata/gate_driver_feature_cols.json. Run training.py first.")

    if not os.path.exists(label_mapping_path):
        raise FileNotFoundError("Missing modeldata/gate_driver_label_mapping.json. Run training.py first.")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing modeldata/gate_driver_scaler.pkl. Run training.py first.")

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    with open(label_mapping_path, "r") as f:
        label_to_id = json.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = pd.read_excel(test_file)

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in test file.")

    missing_cols = [c for c in feature_cols if c not in df.columns]

    if len(missing_cols) > 0:
        raise ValueError(f"Test file missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    X = scaler.transform(X).astype(np.float32)

    input_dim = len(feature_cols)

    if model_name == "CNNMNIST":
        print("Converting gate-driver export/test features to CNNMNIST input shape [N, 1, 16, 16]...")
        X = tabular_to_cnnmnist_image(X)

    y_raw = df[label_col].astype(str)
    unknown = set(y_raw.unique()) - set(label_to_id.keys())

    if len(unknown) > 0:
        raise ValueError(f"Test file has labels not seen during training: {unknown}")

    y = y_raw.map(label_to_id).values.astype(np.int64)

    test_data = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

    num_classes = len(label_to_id)

    print("Gate-driver TEST Excel:", test_file)
    print("Samples:", len(test_data))
    print("Original input dimension:", input_dim)
    print("Classes:", label_to_id)

    return test_data, num_classes, input_dim


def export_to_hfile(quantized_model, filename, runname, modelname="", input_dim=None, num_classes=None):
    if not quantized_model.quantized_model:
        raise ValueError("quantized_model is empty or None")

    max_n_activations = max([
        layer["incoming_weights"]
        for layer in quantized_model.quantized_model
        if "incoming_weights" in layer
    ])

    with open(filename, "w") as f:
        f.write("// Automatically generated header file\n")
        f.write(f"// Date: {datetime.now()}\n")
        f.write(f"// Quantized model exported from {runname}.pth\n")
        f.write("// Generated by exportquant.py\n\n")

        f.write("#include <stdint.h>\n\n")
        f.write("#ifndef BITNETMCU_MODEL_H\n")
        f.write("#define BITNETMCU_MODEL_H\n\n")

        f.write("// Model class name\n")
        f.write(f"#define MODEL_{modelname}\n\n")

        if input_dim is not None:
            f.write(f"#define MODEL_INPUT_DIM {input_dim}\n")

        if num_classes is not None:
            f.write(f"#define MODEL_NUM_CLASSES {num_classes}\n\n")

        f.write(f"#define NUM_LAYERS {len(quantized_model.quantized_model)}\n")
        f.write(f"#define MAX_N_ACTIVATIONS {max_n_activations}\n\n")

        for layer_info in quantized_model.quantized_model:
            layer = f'L{layer_info["layer_order"]}'

            if layer_info["layer_type"] == "BitLinear":
                incoming_weights = layer_info["incoming_weights"]
                outgoing_weights = layer_info["outgoing_weights"]
                bpw = layer_info["bpw"]
                weights = np.array(layer_info["quantized_weights"])
                quantization_type = layer_info["quantization_type"]

                if (bpw * incoming_weights % 32) != 0:
                    raise ValueError(
                        f"Size mismatch: incoming weights must pack to 32-bit boundary. "
                        f"Layer={layer}, incoming={incoming_weights}, bpw={bpw}, bits={bpw * incoming_weights}."
                    )

                print(
                    f"Layer: {layer} Quantization type: <{quantization_type}>, "
                    f"Bits per weight: {bpw}, Incoming: {incoming_weights}, Outgoing: {outgoing_weights}"
                )

                data_type = np.uint32

                if quantization_type in ("Binary", "BinaryBalanced"):
                    encoded_weights = np.where(weights < 0, 0, 1).astype(data_type)
                    QuantID = 1

               # elif quantization_type == "Binary":
                  #  encoded_weights = np.where(weights == -1, 0, 1)
                   # QuantID = 1

                elif quantization_type == "2bitsym":
                    encoded_weights = ((weights < 0).astype(data_type) << 1) | (
                        np.floor(np.abs(weights))
                    ).astype(data_type)
                    QuantID = 2

                elif quantization_type == "4bitsym":
                    encoded_weights = ((weights < 0).astype(data_type) << 3) | (
                        np.floor(np.abs(weights))
                    ).astype(data_type)
                    QuantID = 4

                elif quantization_type == "4bit":
                    encoded_weights = np.floor(weights).astype(data_type) & 15
                    QuantID = 8 + 4

                elif quantization_type == "8bit":
                    encoded_weights = np.floor(weights).astype(data_type) & 255
                    QuantID = 8 + 8

                elif quantization_type == "NF4":
                    levels = np.array([
                        -1.0, -0.6962, -0.5251, -0.3949,
                        -0.2844, -0.1848, -0.0911, 0.0,
                        0.0796, 0.1609, 0.2461, 0.3379,
                        0.4407, 0.5626, 0.723, 1.0,
                    ])
                    encoded_weights = np.argmin(
                        np.abs(weights[:, :, np.newaxis] - levels),
                        axis=2,
                    )
                    QuantID = 32 + 4

                elif quantization_type == "FP130":
                    encoded_weights = ((weights < 0).astype(data_type) << 3) | (
                        np.floor(np.log2(np.abs(weights)))
                    ).astype(data_type)
                    QuantID = 16 + 4

                elif quantization_type == "Ternary":
                    n_outputs, n_inputs = weights.shape

                    if n_inputs % 10 != 0:
                        pad_size = 10 - (n_inputs % 10)
                        print(
                            f"WARNING: Ternary layer {layer} has {n_inputs} inputs, "
                            f"padding with {pad_size} zeros."
                        )
                        weights = np.pad(
                            weights,
                            ((0, 0), (0, pad_size)),
                            mode="constant",
                            constant_values=0,
                        )
                        n_inputs = weights.shape[1]

                    trit_values = np.where(
                        weights == 1,
                        0,
                        np.where(weights == -1, 1, 2),
                    ).astype(np.uint32)

                    packed_row_size = n_inputs // 10
                    packed_weights = np.zeros((n_outputs, packed_row_size), dtype=np.uint16)

                    for row in range(n_outputs):
                        for word_idx in range(packed_row_size):
                            start = word_idx * 10
                            chunk = trit_values[row, start:start + 10]

                            value = 0
                            for t in range(10):
                                value = value * 3 + chunk[t]

                            packed = (value * 65536 + 59048) // 59049
                            packed_weights[row, word_idx] = packed

                    QuantID = 64

                    f.write(f"// Layer: {layer}\n")
                    f.write(f"// QuantType: {quantization_type}\n")
                    f.write(f"#define {layer}_active\n")
                    f.write(f"#define {layer}_bitperweight {QuantID}\n")
                    f.write(f"#define {layer}_incoming_weights {n_inputs}\n")
                    f.write(f"#define {layer}_outgoing_weights {outgoing_weights}\n")

                    f.write(f"const uint16_t {layer}_weights[] = {{")
                    for i, data in enumerate(packed_weights.flatten()):
                        if i % 10 == 0:
                            f.write("\n\t")
                        f.write(f"0x{data:04x},")
                    f.write("\n};\n\n")

                    continue

                else:
                    print(f"Skipping layer {layer}. Unsupported quantization type: {quantization_type}")
                    continue

                weight_per_word = 32 // bpw
                reshaped_array = encoded_weights.reshape(-1, weight_per_word)

                bit_positions = 32 - bpw - np.arange(weight_per_word, dtype=data_type) * bpw

                packed_weights = np.bitwise_or.reduce(
                    reshaped_array << bit_positions,
                    axis=1,
                ).view(data_type)

                f.write(f"// Layer: {layer}\n")
                f.write(f"// QuantType: {quantization_type}\n")
                f.write(f"#define {layer}_active\n")
                f.write(f"#define {layer}_bitperweight {QuantID}\n")
                f.write(f"#define {layer}_incoming_weights {incoming_weights}\n")
                f.write(f"#define {layer}_outgoing_weights {outgoing_weights}\n")

                f.write(f"const uint32_t {layer}_weights[] = {{")
                for i, data in enumerate(packed_weights.flatten()):
                    if i & 7 == 0:
                        f.write("\n\t")
                    f.write(f"0x{data:08x},")
                f.write("\n}; // first channel is topmost bit\n\n")

            elif layer_info["layer_type"] == "BitConv2d":
                in_channels = layer_info["in_channels"]
                out_channels = layer_info["out_channels"]
                incoming_x = layer_info["incoming_x"]
                incoming_y = layer_info["incoming_y"]
                outgoing_x = layer_info["outgoing_x"]
                outgoing_y = layer_info["outgoing_y"]
                groups = layer_info["groups"]
                kernel_size = layer_info["kernel_size"][0]
                bpw = layer_info["bpw"]
                weights = np.array(layer_info["quantized_weights"])

                f.write(f"// Layer: {layer} Convolutional\n")
                f.write(f"#define {layer}_active\n")
                f.write(f"#define {layer}_type BitConv2d\n")
                f.write(f"#define {layer}_in_channels {in_channels}\n")
                f.write(f"#define {layer}_out_channels {out_channels}\n")
                f.write(f"#define {layer}_incoming_x {incoming_x}\n")
                f.write(f"#define {layer}_incoming_y {incoming_y}\n")
                f.write(f"#define {layer}_outgoing_x {outgoing_x}\n")
                f.write(f"#define {layer}_outgoing_y {outgoing_y}\n")
                f.write(f"#define {layer}_kernel_size {kernel_size}\n")
                f.write(f"#define {layer}_stride 1\n")
                f.write(f"#define {layer}_padding 0\n")
                f.write(f"#define {layer}_groups {groups}\n")
                f.write(f"#define {layer}_bitperweight {bpw}\n")

                f.write(f"const int8_t {layer}_weights[] = {{")
                for i, data in enumerate(weights.flatten()):
                    if i % 16 == 0:
                        f.write("\n\t")
                    f.write(f"{int(data)},")
                f.write("\n};\n\n")

            elif layer_info["layer_type"] == "MaxPool2d":
                pool_size = layer_info["kernel_size"]
                incoming_x = layer_info["incoming_x"]
                incoming_y = layer_info["incoming_y"]
                outgoing_x = layer_info["outgoing_x"]
                outgoing_y = layer_info["outgoing_y"]

                f.write(f"#define {layer}_active\n")
                f.write(f"#define {layer}_type MaxPool2d\n")
                f.write(f"#define {layer}_pool_size {pool_size}\n")
                f.write(f"#define {layer}_incoming_x {incoming_x}\n")
                f.write(f"#define {layer}_incoming_y {incoming_y}\n")
                f.write(f"#define {layer}_outgoing_x {outgoing_x}\n")
                f.write(f"#define {layer}_outgoing_y {outgoing_y}\n\n")

        f.write("#endif\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export quantized BitNetMCU model")
    parser.add_argument(
        "--params",
        type=str,
        help="Name of parameter YAML file",
        default="trainingparameters.yaml",
    )

    args = parser.parse_args()
    paramname = args.params

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    hyperparameters.setdefault("augmentation", False)
    hyperparameters.setdefault("network_width3", 0)
    hyperparameters.setdefault("dropout", 0.0)

    runname = create_run_name(hyperparameters)
    print("Run name:", runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = hyperparameters.get("dataset", "FACE").upper()

    if dataset_name != "FACE":
        raise ValueError("This exportquant.py is prepared for dataset: FACE only.")

    test_data, num_classes, input_dim = load_gate_driver_test_excel(
        test_file=hyperparameters["test_file"],
        label_col=hyperparameters.get("label_col", "label"),
        model_name=hyperparameters["model"],
    )

    hyperparameters["num_classes"] = num_classes
    hyperparameters["input_dim"] = input_dim

    model = load_model(hyperparameters["model"], hyperparameters).to(device)

    if hyperparameters["model"] == "CNNMNIST":
        dummy_input = torch.randn(1, 1, 16, 16).to(device)
    else:
        dummy_input = torch.randn(1, input_dim).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print("Model output shape:", dummy_output.shape)
    print("Expected classes:", num_classes)

    if dummy_output.shape[1] != num_classes:
        raise ValueError(
            f"Model output mismatch. Model outputs {dummy_output.shape[1]} classes, "
            f"but dataset needs {num_classes} classes."
        )

    model_path = f"modeldata/{runname}.pth"
    best_model_path = f"modeldata/{runname}_best.pth"

    if os.path.exists(best_model_path):
        model_path = best_model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Cannot find model checkpoint:\n"
            f"{model_path}\n"
            f"Expected either:\n"
            f"modeldata/{runname}.pth\n"
            f"or\n"
            f"modeldata/{runname}_best.pth"
        )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cpu"),
        )
    )

    print("Loaded checkpoint:", model_path)

    model = model.to(device)
    model.eval()

    print("Quantizing model...")
    quantized_model = QuantizedModel(model)

    print(f"Total number of bits: {quantized_model.totalbits()}")
    print(f"Total size: {quantized_model.totalbits() / 8 / 1024:.4f} KB")

    export_header = hyperparameters.get("export_header", "BitNetMCU_model.h")

    print("Exporting model to header file:", export_header)

    export_to_hfile(
        quantized_model,
        export_header,
        runname,
        hyperparameters["model"],
        input_dim=256 if hyperparameters["model"] == "CNNMNIST" else input_dim,
        num_classes=num_classes,
    )

    print("Export done:", export_header)
