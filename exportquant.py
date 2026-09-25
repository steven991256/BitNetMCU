import os
import argparse
from datetime import datetime

import yaml
import numpy as np

import torch
from torchvision import datasets, transforms

from BitNetMCU import QuantizedModel


# ================================================================
# Run name
# ================================================================

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


# ================================================================
# Load model
# ================================================================

def load_model(model_name, params):

    import importlib

    module = importlib.import_module("models")
    model_class = getattr(module, model_name)

    kwargs = {
        "network_width1": params["network_width1"],
        "network_width2": params["network_width2"],
        "network_width3": params["network_width3"],
        "QuantType": params["QuantType"],
        "NormType": params["NormType"],
        "WScale": params["WScale"],
    }

    if "cnn_width" in params:
        kwargs["cnn_width"] = params["cnn_width"]

    if "num_classes" in params:
        kwargs["num_classes"] = params["num_classes"]

    return model_class(**kwargs)


# ================================================================
# Evaluate original PyTorch model
# ================================================================

def evaluate_model(model, test_data, device):

    model.eval()

    correct = 0
    total = 0

    predictions = []
    labels_list = []

    with torch.no_grad():

        for image, label in test_data:

            image = image.unsqueeze(0).to(device)

            output = model(image)

            prediction = int(torch.argmax(output, dim=1).item())

            predictions.append(prediction)
            labels_list.append(int(label))

            if prediction == label:
                correct += 1

            total += 1

    accuracy = 100.0 * correct / total

    return accuracy, predictions, labels_list


# ================================================================
# Encode weights
#
# IMPORTANT:
# This keeps the BitNetMCU encoding used by your original
# exporter. We are NOT changing the C/FPGA weight format here.
# ================================================================

def encode_linear_weights(weights, quantization_type, data_type):

    weights = np.asarray(weights)

    # ------------------------------------------------------------
    # Binary
    # ------------------------------------------------------------

    if quantization_type in ("Binary", "BinaryBalanced"):

        encoded_weights = np.where(
            weights < 0,
            0,
            1
        ).astype(data_type)

        quant_id = 1

    # ------------------------------------------------------------
    # 2-bit symmetric
    # ------------------------------------------------------------

    elif quantization_type == "2bitsym":

        magnitude = np.floor(
            np.abs(weights)
        ).astype(data_type)

        sign = (
            (weights < 0)
            .astype(data_type)
            << 1
        )

        encoded_weights = sign | magnitude

        quant_id = 2

    # ------------------------------------------------------------
    # 4-bit symmetric
    # ------------------------------------------------------------

    elif quantization_type == "4bitsym":

        magnitude = np.floor(
            np.abs(weights)
        ).astype(data_type)

        sign = (
            (weights < 0)
            .astype(data_type)
            << 3
        )

        encoded_weights = sign | magnitude

        quant_id = 4

    # ------------------------------------------------------------
    # 4-bit
    # ------------------------------------------------------------

    elif quantization_type == "4bit":

        # QuantizedModel should already have quantized values.
        # Do NOT perform another floating-point quantization.
        #
        # We only convert the already-quantized values into
        # their 4-bit storage representation.

        rounded = np.asarray(weights)

        # Verify that values are effectively integral.
        if not np.allclose(
            rounded,
            np.round(rounded),
            atol=1e-5
        ):
            raise ValueError(
                "4bit weights contain non-integer values. "
                "The exporter received values that appear to "
                "still be floating-point/continuous."
            )

        rounded = np.round(rounded).astype(data_type)

        # Keep lower 4 bits.
        encoded_weights = rounded & 0xF

        # BitNetMCU QuantID from original exporter.
        quant_id = 8 + 4

    # ------------------------------------------------------------
    # 8-bit
    # ------------------------------------------------------------

    elif quantization_type == "8bit":

        rounded = np.asarray(weights)

        if not np.allclose(
            rounded,
            np.round(rounded),
            atol=1e-5
        ):
            raise ValueError(
                "8bit weights contain non-integer values."
            )

        encoded_weights = (
            np.round(rounded)
            .astype(data_type)
            & 0xFF
        )

        quant_id = 8 + 8

    # ------------------------------------------------------------
    # NF4
    # ------------------------------------------------------------

    elif quantization_type == "NF4":

        levels = np.array([
            -1.0,
            -0.6962,
            -0.5251,
            -0.3949,
            -0.2844,
            -0.1848,
            -0.0911,
             0.0,
             0.0796,
             0.1609,
             0.2461,
             0.3379,
             0.4407,
             0.5626,
             0.7230,
             1.0,
        ])

        encoded_weights = np.argmin(
            np.abs(
                weights[:, :, np.newaxis]
                - levels
            ),
            axis=2
        ).astype(data_type)

        quant_id = 32 + 4

    # ------------------------------------------------------------
    # FP130
    # ------------------------------------------------------------

    elif quantization_type == "FP130":

        abs_weights = np.abs(weights)

        encoded_weights = (
            (
                (weights < 0)
                .astype(data_type)
                << 3
            )
            |
            np.floor(
                np.log2(
                    np.maximum(abs_weights, 1e-12)
                )
            ).astype(data_type)
        )

        quant_id = 16 + 4

    else:

        raise ValueError(
            f"Unsupported linear quantization type: "
            f"{quantization_type}"
        )

    return encoded_weights, quant_id


# ================================================================
# Pack weights into 32-bit words
# ================================================================

def pack_weights_32bit(encoded_weights, bpw):

    if bpw <= 0:
        raise ValueError(
            f"Invalid bits-per-weight: {bpw}"
        )

    if 32 % bpw != 0:
        raise ValueError(
            f"Bits-per-weight {bpw} does not divide "
            f"a 32-bit word."
        )

    weight_per_word = 32 // bpw

    flat = np.asarray(
        encoded_weights,
        dtype=np.uint32
    ).flatten()

    if len(flat) % weight_per_word != 0:

        raise ValueError(
            f"Cannot pack {len(flat)} weights at "
            f"{bpw} bits/weight."
        )

    reshaped = flat.reshape(
        -1,
        weight_per_word
    )

    packed = np.zeros(
        reshaped.shape[0],
        dtype=np.uint32
    )

    # First weight occupies the top bits.
    for i in range(weight_per_word):

        shift = (
            32
            - bpw
            - i * bpw
        )

        packed |= (
            reshaped[:, i]
            << shift
        )

    return packed


# ================================================================
# Export header
# ================================================================

def export_to_hfile(
    quantized_model,
    filename,
    runname,
    modelname="",
    input_dim=None,
    num_classes=None,
):

    if not quantized_model.quantized_model:

        raise ValueError(
            "Quantized model is empty."
        )

    layers = quantized_model.quantized_model

    # ------------------------------------------------------------
    # Maximum activation/weight input size
    # ------------------------------------------------------------

    incoming_values = [
        layer["incoming_weights"]
        for layer in layers
        if "incoming_weights" in layer
    ]

    if not incoming_values:
        raise ValueError(
            "No layers with incoming_weights found."
        )

    max_n_activations = max(
        incoming_values
    )

    print()
    print("=" * 60)
    print("EXPORTING QUANTIZED MODEL")
    print("=" * 60)

    print(
        f"Number of layers : {len(layers)}"
    )

    print(
        f"Max activations  : {max_n_activations}"
    )

    print(
        f"Quantized bits   : "
        f"{quantized_model.totalbits()}"
    )

    print(
        f"Quantized bytes  : "
        f"{quantized_model.totalbits() / 8:.0f}"
    )

    print(
        f"Quantized size   : "
        f"{quantized_model.totalbits() / 8 / 1024:.4f} KB"
    )

    print("=" * 60)

    with open(filename, "w") as f:

        f.write(
            "// Automatically generated header file\n"
        )

        f.write(
            f"// Date: {datetime.now()}\n"
        )

        f.write(
            f"// Quantized model exported from "
            f"{runname}.pth\n"
        )

        f.write(
            "// Generated by corrected exportquant.py\n\n"
        )

        f.write(
            "#include <stdint.h>\n\n"
        )

        f.write(
            "#ifndef BITNETMCU_MODEL_H\n"
        )

        f.write(
            "#define BITNETMCU_MODEL_H\n\n"
        )

        f.write(
            f"#define MODEL_{modelname}\n\n"
        )

        if input_dim is not None:

            f.write(
                f"#define MODEL_INPUT_DIM "
                f"{input_dim}\n"
            )

        if num_classes is not None:

            f.write(
                f"#define MODEL_NUM_CLASSES "
                f"{num_classes}\n\n"
            )

        f.write(
            f"#define NUM_LAYERS "
            f"{len(layers)}\n"
        )

        f.write(
            f"#define MAX_N_ACTIVATIONS "
            f"{max_n_activations}\n\n"
        )

        # ========================================================
        # Layers
        # ========================================================

        for layer_info in layers:

            layer = (
                f'L{layer_info["layer_order"]}'
            )

            layer_type = (
                layer_info["layer_type"]
            )

            print()
            print(
                f"Exporting {layer}: "
                f"{layer_type}"
            )

            # ====================================================
            # BitLinear
            # ====================================================

            if layer_type == "BitLinear":

                incoming_weights = int(
                    layer_info["incoming_weights"]
                )

                outgoing_weights = int(
                    layer_info["outgoing_weights"]
                )

                bpw = int(
                    layer_info["bpw"]
                )

                weights = np.asarray(
                    layer_info["quantized_weights"]
                )

                quantization_type = (
                    layer_info["quantization_type"]
                )

                print(
                    f"  QuantType : "
                    f"{quantization_type}"
                )

                print(
                    f"  Shape     : "
                    f"{weights.shape}"
                )

                print(
                    f"  BPW       : {bpw}"
                )

                print(
                    f"  Range     : "
                    f"{weights.min()} "
                    f"to {weights.max()}"
                )

                expected_shape = (
                    outgoing_weights,
                    incoming_weights
                )

                if weights.shape != expected_shape:

                    raise ValueError(
                        f"{layer}: weight shape "
                        f"{weights.shape} does not match "
                        f"expected "
                        f"{expected_shape}"
                    )

                if (
                    bpw * incoming_weights
                ) % 32 != 0:

                    raise ValueError(
                        f"{layer}: incoming weights "
                        f"cannot be packed into 32-bit "
                        f"words."
                    )

                encoded_weights, quant_id = (
                    encode_linear_weights(
                        weights,
                        quantization_type,
                        np.uint32
                    )
                )

                packed_weights = pack_weights_32bit(
                    encoded_weights,
                    bpw
                )

                f.write(
                    f"// Layer: {layer}\n"
                )

                f.write(
                    f"// Layer type: BitLinear\n"
                )

                f.write(
                    f"// QuantType: "
                    f"{quantization_type}\n"
                )

                f.write(
                    f"#define {layer}_active\n"
                )

                f.write(
                    f"#define {layer}_bitperweight "
                    f"{quant_id}\n"
                )

                f.write(
                    f"#define {layer}_incoming_weights "
                    f"{incoming_weights}\n"
                )

                f.write(
                    f"#define {layer}_outgoing_weights "
                    f"{outgoing_weights}\n"
                )

                f.write(
                    f"const uint32_t "
                    f"{layer}_weights[] = {{"
                )

                for i, data in enumerate(
                    packed_weights
                ):

                    if i % 8 == 0:

                        f.write(
                            "\n\t"
                        )

                    f.write(
                        f"0x{int(data):08x},"
                    )

                f.write(
                    "\n};\n\n"
                )

            # ====================================================
            # BitConv2d
            # ====================================================

            elif layer_type == "BitConv2d":

                in_channels = int(
                    layer_info["in_channels"]
                )

                out_channels = int(
                    layer_info["out_channels"]
                )

                groups = int(
                    layer_info["groups"]
                )

                kernel_size = int(
                    layer_info["kernel_size"][0]
                )

                bpw = int(
                    layer_info["bpw"]
                )

                weights = np.asarray(
                    layer_info["quantized_weights"]
                )

                layer_order = int(
                    layer_info["layer_order"]
                )

                # ------------------------------------------------
                # CNN dimensions
                # ------------------------------------------------

                if layer_order == 2:

                    incoming_x = 32
                    incoming_y = 32

                    outgoing_x = 30
                    outgoing_y = 30

                elif layer_order == 4:

                    incoming_x = 30
                    incoming_y = 30

                    outgoing_x = 28
                    outgoing_y = 28

                elif layer_order == 7:

                    incoming_x = 14
                    incoming_y = 14

                    outgoing_x = 12
                    outgoing_y = 12

                else:

                    raise ValueError(
                        f"Unexpected BitConv2d "
                        f"layer order {layer_order}. "
                        f"Expected 2, 4, or 7."
                    )

                print(
                    f"  Shape     : "
                    f"{weights.shape}"
                )

                print(
                    f"  BPW       : {bpw}"
                )

                print(
                    f"  Range     : "
                    f"{weights.min()} "
                    f"to {weights.max()}"
                )

                # ------------------------------------------------
                # Header
                # ------------------------------------------------

                f.write(
                    f"// Layer: {layer}\n"
                )

                f.write(
                    "// Layer type: BitConv2d\n"
                )

                f.write(
                    "#define "
                    f"{layer}_active\n"
                )

                f.write(
                    f"#define {layer}_type "
                    f"BitConv2d\n"
                )

                f.write(
                    f"#define {layer}_in_channels "
                    f"{in_channels}\n"
                )

                f.write(
                    f"#define {layer}_out_channels "
                    f"{out_channels}\n"
                )

                f.write(
                    f"#define {layer}_incoming_x "
                    f"{incoming_x}\n"
                )

                f.write(
                    f"#define {layer}_incoming_y "
                    f"{incoming_y}\n"
                )

                f.write(
                    f"#define {layer}_outgoing_x "
                    f"{outgoing_x}\n"
                )

                f.write(
                    f"#define {layer}_outgoing_y "
                    f"{outgoing_y}\n"
                )

                f.write(
                    f"#define {layer}_kernel_size "
                    f"{kernel_size}\n"
                )

                f.write(
                    f"#define {layer}_stride 1\n"
                )

                f.write(
                    f"#define {layer}_padding 0\n"
                )

                f.write(
                    f"#define {layer}_groups "
                    f"{groups}\n"
                )

                f.write(
                    f"#define {layer}_bitperweight "
                    f"{bpw}\n"
                )

                f.write(
                    f"const int8_t "
                    f"{layer}_weights[] = {{"
                )

                for i, data in enumerate(
                    weights.flatten()
                ):

                    if i % 16 == 0:

                        f.write(
                            "\n\t"
                        )

                    f.write(
                        f"{int(data)},"
                    )

                f.write(
                    "\n};\n\n"
                )

            # ====================================================
            # MaxPool2d
            # ====================================================

            elif layer_type == "MaxPool2d":

                pool_size = int(
                    layer_info["kernel_size"]
                )

                layer_order = int(
                    layer_info["layer_order"]
                )

                if layer_order == 6:

                    incoming_x = 28
                    incoming_y = 28

                    outgoing_x = 14
                    outgoing_y = 14

                elif layer_order == 9:

                    incoming_x = 12
                    incoming_y = 12

                    outgoing_x = 6
                    outgoing_y = 6

                else:

                    raise ValueError(
                        f"Unexpected MaxPool2d "
                        f"layer order {layer_order}. "
                        f"Expected 6 or 9."
                    )

                f.write(
                    f"// Layer: {layer}\n"
                )

                f.write(
                    "// Layer type: MaxPool2d\n"
                )

                f.write(
                    f"#define {layer}_active\n"
                )

                f.write(
                    f"#define {layer}_type "
                    f"MaxPool2d\n"
                )

                f.write(
                    f"#define {layer}_pool_size "
                    f"{pool_size}\n"
                )

                f.write(
                    f"#define {layer}_incoming_x "
                    f"{incoming_x}\n"
                )

                f.write(
                    f"#define {layer}_incoming_y "
                    f"{incoming_y}\n"
                )

                f.write(
                    f"#define {layer}_outgoing_x "
                    f"{outgoing_x}\n"
                )

                f.write(
                    f"#define {layer}_outgoing_y "
                    f"{outgoing_y}\n\n"
                )

            # ====================================================
            # Unknown layer
            # ====================================================

            else:

                raise ValueError(
                    f"Unsupported layer type: "
                    f"{layer_type}"
                )

        f.write(
            "#endif\n"
        )

    print()
    print(
        f"Header successfully written: {filename}"
    )


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Export BitNetMCU quantized FACE CNN "
            "model to C header"
        )
    )

    parser.add_argument(
        "--params",
        type=str,
        default="trainingparameters.yaml",
        help="YAML parameter file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final",
        choices=["final", "best"],
        help=(
            "Checkpoint to export. "
            "Default: final"
        )
    )

    args = parser.parse_args()

    # ============================================================
    # Parameters
    # ============================================================

    print(
        f"Loading parameters from: "
        f"{args.params}"
    )

    with open(args.params) as f:

        hyperparameters = yaml.safe_load(f)

    hyperparameters.setdefault(
        "augmentation",
        False
    )

    hyperparameters.setdefault(
        "network_width3",
        0
    )

    hyperparameters.setdefault(
        "dropout",
        0.0
    )

    runname = create_run_name(
        hyperparameters
    )

    print()
    print(
        f"Run name: {runname}"
    )

    # ============================================================
    # Device
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ============================================================
    # Dataset
    # ============================================================

    dataset_name = (
        hyperparameters
        .get("dataset", "FACE")
        .upper()
    )

    if dataset_name != "FACE":

        raise ValueError(
            "This exporter is prepared "
            "for FACE dataset only."
        )

    data_root = hyperparameters[
        "data_root"
    ]

    test_dir = os.path.join(
        data_root,
        hyperparameters["test_folder"]
    )

    transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Resize(
            (32, 32)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5,),
            (0.5,)
        ),
    ])

    test_data = datasets.ImageFolder(
        test_dir,
        transform=transform
    )

    num_classes = len(
        test_data.classes
    )

    input_dim = 32 * 32

    print()
    print("=" * 60)
    print("FACE DATASET")
    print("=" * 60)

    print(
        f"Test directory : {test_dir}"
    )

    print(
        f"Test samples   : {len(test_data)}"
    )

    print(
        f"Number classes : {num_classes}"
    )

    print(
        f"Class mapping  : "
        f"{test_data.class_to_idx}"
    )

    print("=" * 60)

    hyperparameters[
        "num_classes"
    ] = num_classes

    hyperparameters[
        "input_dim"
    ] = input_dim

    # ============================================================
    # Create model
    # ============================================================

    model = load_model(
        hyperparameters["model"],
        hyperparameters
    ).to(device)

    # ============================================================
    # Check model output
    # ============================================================

    if hyperparameters["model"] == "CNNMNIST":

        dummy_input = torch.randn(
            1,
            1,
            32,
            32,
            device=device
        )

    else:

        dummy_input = torch.randn(
            1,
            input_dim,
            device=device
        )

    with torch.no_grad():

        dummy_output = model(
            dummy_input
        )

    print()
    print(
        f"Model input shape  : "
        f"{dummy_input.shape}"
    )

    print(
        f"Model output shape : "
        f"{dummy_output.shape}"
    )

    if (
        dummy_output.ndim != 2
        or dummy_output.shape[1]
        != num_classes
    ):

        raise ValueError(
            "Model output does not match "
            f"the {num_classes} classes."
        )

    # ============================================================
    # Select checkpoint
    # ============================================================

    final_checkpoint = (
        f"modeldata/{runname}.pth"
    )

    best_checkpoint = (
        f"modeldata/{runname}_best.pth"
    )

    if args.checkpoint == "best":

        model_path = best_checkpoint

    else:

        model_path = final_checkpoint

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{model_path}\n\n"
            f"Final checkpoint:\n"
            f"{final_checkpoint}\n\n"
            f"Best checkpoint:\n"
            f"{best_checkpoint}"
        )

    # ============================================================
    # Load checkpoint
    # ============================================================

    print()
    print("=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)

    print(
        f"Checkpoint mode : "
        f"{args.checkpoint}"
    )

    print(
        f"Checkpoint path : "
        f"{model_path}"
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu"
    )

    # Handle either a raw state_dict or a
    # checkpoint dictionary.

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(device)
    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # ============================================================
    # IMPORTANT:
    # Test the ORIGINAL PyTorch model BEFORE quantization.
    # ============================================================

    print()
    print("=" * 60)
    print("ORIGINAL PYTORCH MODEL")
    print("=" * 60)

    original_accuracy, _, _ = (
        evaluate_model(
            model,
            test_data,
            device
        )
    )

    print(
        f"Original model accuracy: "
        f"{original_accuracy:.2f}%"
    )

    print(
        "This is the reference accuracy "
        "for this exact checkpoint and "
        "this exact test set."
    )

    # ============================================================
    # Quantize
    # ============================================================

    print()
    print("=" * 60)
    print("CREATING BITNETMCU QUANTIZED MODEL")
    print("=" * 60)

    quantized_model = QuantizedModel(
        model
    )

    total_bits = (
        quantized_model.totalbits()
    )

    total_bytes = (
        total_bits / 8
    )

    total_kb = (
        total_bytes / 1024
    )

    print(
        f"Total bits  : {total_bits}"
    )

    print(
        f"Total bytes : {total_bytes:.0f}"
    )

    print(
        f"Total KB    : {total_kb:.4f}"
    )

    # ============================================================
    # Verify expected 4-bit configuration
    # ============================================================

    requested_quant_type = (
        hyperparameters["QuantType"]
    )

    print()
    print(
        f"Requested QuantType: "
        f"{requested_quant_type}"
    )

    if requested_quant_type == "4bit":

        print(
            "4-bit export checks enabled."
        )

        for layer_info in (
            quantized_model.quantized_model
        ):

            if (
                layer_info["layer_type"]
                == "BitLinear"
            ):

                qtype = (
                    layer_info[
                        "quantization_type"
                    ]
                )

                bpw = int(
                    layer_info["bpw"]
                )

                if qtype != "4bit":

                    raise ValueError(
                        "Expected BitLinear "
                        f"4bit layer but found "
                        f"{qtype}."
                    )

                if bpw != 4:

                    raise ValueError(
                        f"Expected 4 bits/weight "
                        f"but layer has "
                        f"{bpw}."
                    )

    # ============================================================
    # Export
    # ============================================================

    export_header = (
        hyperparameters.get(
            "export_header",
            "BitNetMCU_model.h"
        )
    )

    print()
    print("=" * 60)
    print("EXPORT")
    print("=" * 60)

    print(
        f"Output header: "
        f"{export_header}"
    )

    export_to_hfile(
        quantized_model,
        export_header,
        runname,
        hyperparameters["model"],
        input_dim=(
            1024
            if hyperparameters["model"]
            == "CNNMNIST"
            else input_dim
        ),
        num_classes=num_classes
    )

    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)

    print(
        f"Checkpoint : {model_path}"
    )

    print(
        f"Original accuracy : "
        f"{original_accuracy:.2f}%"
    )

    print(
        f"Model size : "
        f"{total_kb:.4f} KB"
    )

    print(
        f"Header : "
        f"{export_header}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The original PyTorch accuracy above "
        "is the reference. If this accuracy "
        "is already 53.33%, the export itself "
        "cannot be blamed for that accuracy "
        "drop."
    )

    print(
        "Use --checkpoint best only when you "
        "have verified that _best.pth contains "
        "the intended 80% validation checkpoint."
    )

