import os
import time
import json
import pickle
import random
import argparse
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score

from BitNetMCU import BitLinear, BitConv2d, Activation

try:
    from torchsummary import summary
except Exception:
    summary = None

try:
    import importlib
    from models import MaskingLayer
except Exception:
    MaskingLayer = None
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
            print("Warning: BitLinear argument mismatch. Falling back to nn.Linear.")
            return nn.Linear(in_features, out_features)


class GateDriverMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        network_width1=64,
        network_width2=32,
        network_width3=0,
        QuantType="BinaryBalanced",
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

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(make_bitlinear(network_width1, network_width2, QuantType, NormType, WScale))
        layers.append(Activation())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        if network_width3 and network_width3 > 0:
            layers.append(make_bitlinear(network_width2, network_width3, QuantType, NormType, WScale))
            layers.append(Activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

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
        raise ValueError(f"Model {model_name} not found in models.py or training.py")


def tabular_to_cnnmnist_image(X):
    """
    Convert tabular gate-driver features into CNNMNIST input.

    Input:
        X shape = [N, num_features]

    Output:
        X shape = [N, 1, 16, 16]
    """
    if X.shape[1] > 256:
        raise ValueError(f"Too many features for 16x16 CNN input: {X.shape[1]} > 256")

    X_pad = np.zeros((X.shape[0], 256), dtype=np.float32)
    X_pad[:, :X.shape[1]] = X
    return X_pad.reshape(-1, 1, 16, 16)


def load_gate_driver_excel(train_file, val_file, label_col="label", model_name="GateDriverMLP"):
    train_df = pd.read_excel(train_file)
    val_df = pd.read_excel(val_file)

    print("Train Excel shape:", train_df.shape)
    print("Validation Excel shape:", val_df.shape)

    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in train file.")

    if label_col not in val_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in validation file.")

    leakage_cols = {
        "label",
        "label_id",
        "sample_id",
        "fault_name",
        "class_name",
        "class",
        "fault",
        "target",
        "y",
        "split",
    }

    feature_cols = []

    for c in train_df.columns:
        c_lower = c.lower()

        if c_lower in leakage_cols:
            continue

        if c_lower.startswith("flag_"):
            continue

        if "label" in c_lower:
            continue

        if "fault" in c_lower and c_lower != "fault_current":
            continue

        if pd.api.types.is_numeric_dtype(train_df[c]):
            feature_cols.append(c)

    if len(feature_cols) == 0:
        raise ValueError("No safe numeric input features found after leakage removal.")

    print("\nSafe input features used by the model:")
    for c in feature_cols:
        print(" -", c)

    train_labels_raw = train_df[label_col].astype(str)
    val_labels_raw = val_df[label_col].astype(str)

    label_names = sorted(train_labels_raw.unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}

    unknown_val_labels = set(val_labels_raw.unique()) - set(label_to_id.keys())
    if len(unknown_val_labels) > 0:
        raise ValueError(f"Validation has labels not found in train: {unknown_val_labels}")

    y_train = train_labels_raw.map(label_to_id).values.astype(np.int64)
    y_val = val_labels_raw.map(label_to_id).values.astype(np.int64)

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    X_train = X_train.values.astype(np.float32)
    X_val = X_val.values.astype(np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    input_dim = len(feature_cols)

    if model_name == "CNNMNIST":
        print("\nConverting tabular gate-driver features to CNNMNIST input shape [N, 1, 16, 16]...")
        X_train = tabular_to_cnnmnist_image(X_train)
        X_val = tabular_to_cnnmnist_image(X_val)

    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_data = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    num_classes = len(label_names)

    os.makedirs("modeldata", exist_ok=True)

    with open("modeldata/gate_driver_feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open("modeldata/gate_driver_label_mapping.json", "w") as f:
        json.dump(label_to_id, f, indent=2)

    with open("modeldata/gate_driver_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nLabel mapping:")
    print(label_to_id)
    print("Original input dimension:", input_dim)
    print("Number of classes:", num_classes)

    return train_data, val_data, num_classes, input_dim, feature_cols


def add_mask_regularization(model, lambda_l1):
    if MaskingLayer is None:
        return 0

    mask_layer = next(
        (layer for layer in model.modules() if isinstance(layer, MaskingLayer)),
        None,
    )

    if mask_layer is None:
        return 0

    return lambda_l1 * torch.norm(mask_layer.mask, 1)


def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
    total_activations = 0
    positive_activations = 0

    def hook_fn(module, input, output):
        nonlocal total_activations, positive_activations

        if isinstance(module, nn.ReLU) or isinstance(module, Activation):
            total_activations += output.numel()
            positive_activations += (output > 0).sum().item()

    hooks = []

    for layer in model.modules():
        if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
            hooks.append(layer.register_forward_hook(hook_fn))

    with torch.no_grad():
        for i in range(0, len(all_test_images), batch_size):
            images = all_test_images[i:i + batch_size]
            model(images)

    for hook in hooks:
        hook.remove()

    if total_activations == 0:
        fraction_positive = 0.0
    else:
        fraction_positive = positive_activations / total_activations

    writer.add_scalar("Activations/positive_fraction", fraction_positive, epoch + 1)
    return fraction_positive


def train_model(model, device, hyperparameters, train_data, val_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname = create_run_name(hyperparameters)
    batch_size = hyperparameters["batch_size"]

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    entire_dataset = next(iter(train_loader))
    all_train_images = entire_dataset[0].to(device)
    all_train_labels = entire_dataset[1].to(device)

    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    entire_val_dataset = next(iter(val_loader))
    all_val_images = entire_val_dataset[0].to(device)
    all_val_labels = entire_val_dataset[1].to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=hyperparameters.get("weight_decay", 0.0),
    )

    scheduler_name = hyperparameters.get("scheduler", "Cosine")

    if scheduler_name == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=hyperparameters.get("step_size", 20),
            gamma=hyperparameters.get("lr_decay", 0.5),
        )

    elif scheduler_name == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=0,
        )

    elif scheduler_name == "CosineWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparameters.get("T_0", 10),
            T_mult=hyperparameters.get("T_mult", 2),
            eta_min=0,
        )

    else:
        raise ValueError("Invalid scheduler")

    criterion = nn.CrossEntropyLoss()

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{runname}-{now_str}")

    best_val_accuracy = 0.0
    best_f1 = 0.0
    totalbits = 0

    for epoch in range(num_epochs):
        model.train()

        correct = 0
        train_loss = []
        start_time = time.time()

        indices = list(range(len(all_train_images)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]

            images = all_train_images[batch_indices]
            labels = all_train_labels[batch_indices]

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            if epoch < hyperparameters.get("prune_epoch", -1):
                loss += add_mask_regularization(model, hyperparameters.get("lambda_l1", 0.0))

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            correct += (predicted == labels).sum().item()

        scheduler.step()

        if epoch + 1 == halve_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5
            print(f"Learning rate halved at epoch {epoch + 1}")

        trainaccuracy = correct / len(train_data) * 100

        model.eval()

        correct = 0
        total = 0
        val_loss = []

        all_preds = []
        all_labels_list = []
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(all_val_images), batch_size):
                images = all_val_images[i:i + batch_size]
                labels = all_val_labels[i:i + batch_size]

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                loss = criterion(outputs, labels)
                val_loss.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        activity = log_positive_activations(
            model,
            writer,
            epoch,
            all_val_images,
            batch_size,
        )

        epoch_time = time.time() - start_time
        valaccuracy = correct / total * 100

        cm = confusion_matrix(all_labels_list, all_preds)

        f1 = f1_score(
            all_labels_list,
            all_preds,
            average="macro",
            zero_division=0,
        )

        print("\nConfusion Matrix:\n", cm)
        print("F1 Score:", f1)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"LTrain:{np.mean(train_loss):.6f} "
            f"ATrain:{trainaccuracy:.2f}% "
            f"LVal:{np.mean(val_loss):.6f} "
            f"AVal:{valaccuracy:.2f}% "
            f"Time[s]:{epoch_time:.2f} "
            f"Act:{activity * 100:.1f}% "
            f"w_clip/entropy[bits]: ",
            end="",
        )

        print("\nClassification Report:\n")
        print(
            classification_report(
                all_labels_list,
                all_preds,
                zero_division=0,
            )
        )

        try:
            roc_auc = roc_auc_score(
                all_labels_list,
                np.array(all_probs),
                multi_class="ovr",
            )
            print("ROC AUC:", roc_auc)
        except Exception:
            print("ROC not available")

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = np.divide(
            TP,
            TP + FN,
            out=np.zeros_like(TP, dtype=float),
            where=(TP + FN) != 0,
        )

        specificity = np.divide(
            TN,
            TN + FP,
            out=np.zeros_like(TN, dtype=float),
            where=(TN + FP) != 0,
        )

        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

        totalbits = 0

        for layer in model.modules():
            if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):
                if epoch < hyperparameters.get("maxw_update_until_epoch", 60):
                    try:
                        layer.update_clipping_scalar(
                            layer.weight,
                            hyperparameters.get("maxw_algo", "octav"),
                            hyperparameters.get("maxw_quantscale", 0.25),
                        )
                    except Exception:
                        pass

                try:
                    w_quant, _, _ = layer.weight_quant(layer.weight)

                    _, counts = np.unique(
                        w_quant.cpu().detach().numpy(),
                        return_counts=True,
                    )

                    probabilities = counts / np.sum(counts)
                    entropy = -np.sum(probabilities * np.log2(probabilities))

                    print(f"{layer.s.item():.3f}/{entropy:.2f}", end=" ")

                    totalbits += layer.weight.numel() * layer.bpw

                except Exception:
                    pass

        print()

        writer.add_scalar("Loss/train", np.mean(train_loss), epoch + 1)
        writer.add_scalar("Accuracy/train", trainaccuracy, epoch + 1)
        writer.add_scalar("Loss/val", np.mean(val_loss), epoch + 1)
        writer.add_scalar("Accuracy/val", valaccuracy, epoch + 1)
        writer.add_scalar("F1/val_macro", f1, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

        if valaccuracy > best_val_accuracy:
            best_val_accuracy = valaccuracy
            best_f1 = f1
            os.makedirs("modeldata", exist_ok=True)
            torch.save(model.state_dict(), f"modeldata/{runname}_best.pth")
            print(f"Best model saved. Best validation accuracy: {best_val_accuracy:.2f}%")

    numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotalBits: {totalbits} TotalBytes: {totalbits / 8.0}")
    print(f"Trainable Parameters: {numofweights}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Best Validation F1 Macro: {best_f1:.4f}")

    writer.add_hparams(
        hyperparameters,
        {
            "Parameters": numofweights,
            "Totalbits": totalbits,
            "Accuracy/train": trainaccuracy,
            "Accuracy/val": valaccuracy,
            "F1/val_macro": f1,
            "Loss/train": np.mean(train_loss),
            "Loss/val": np.mean(val_loss),
        },
    )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BitNetMCU Training Script")
    parser.add_argument(
        "--params",
        type=str,
        help="Name of the parameter YAML file",
        default="trainingparameters.yaml",
    )

    args = parser.parse_args()
    paramname = args.params

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    hyperparameters.setdefault("augmentation", False)
    hyperparameters.setdefault("dropout", 0.0)
    hyperparameters.setdefault("weight_decay", 0.0)
    hyperparameters.setdefault("lambda_l1", 0.0)
    hyperparameters.setdefault("prune_epoch", -1)
    hyperparameters.setdefault("maxw_update_until_epoch", 60)
    hyperparameters.setdefault("maxw_algo", "octav")
    hyperparameters.setdefault("maxw_quantscale", 0.25)
    hyperparameters.setdefault("network_width3", 0)

    runname = create_run_name(hyperparameters)
    print("Run name:", runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

   dataset_name = hyperparameters.get("dataset", "FACE").upper()

   if dataset_name != "FACE":
       raise ValueError("This training.py is prepared for dataset: FACE only.")

   data_root = hyperparameters["data_root"]

   train_dir = os.path.join(data_root, hyperparameters["train_folder"])
   val_dir = os.path.join(data_root, hyperparameters["val_folder"])

   transform = transforms.Compose([
       transforms.Grayscale(num_output_channels=1),
       transforms.Resize((32, 32)),
       transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    num_classes = len(train_data.classes)
    input_dim = 32 * 32

    print("Classes:", train_data.classes)
    print("Training samples:", len(train_data))
    print("Validation samples:", len(val_data))
    print("Input size:", input_dim)

    hyperparameters["num_classes"] = num_classes
    hyperparameters["input_dim"] = input_dim

    model = load_model(hyperparameters["model"], hyperparameters).to(device)

    if hyperparameters["model"] == "CNNMNIST":
        dummy_input = torch.randn(1, 1, 32, 32).to(device)
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

    if summary is not None:
        try:
            if hyperparameters["model"] == "CNNMNIST":
                summary(model, input_size=(1, 32, 32))
            else:
                summary(model, input_size=(input_dim,))
        except Exception as e:
            print("torchsummary skipped:", e)

    print("training...")
    train_model(model, device, hyperparameters, train_data, val_data)

    print("saving final model...")
    os.makedirs("modeldata", exist_ok=True)
    torch.save(model.state_dict(), f"modeldata/{runname}.pth")
    print(f"Saved: modeldata/{runname}.pth")
