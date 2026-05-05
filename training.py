import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from BitNetMCU import BitLinear, BitConv2d, Activation

import time
import random
import argparse
import yaml
import importlib

from torchsummary import summary
from models import MaskingLayer

from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score


# ----------------------------------------------
# Custom signal dataset
# ----------------------------------------------

class SignalDataset(Dataset):
    """
    CSV signal dataset.

    Expected folder structure:

        signal_dataset/split/train/
            idle/
            left/
            right/

        signal_dataset/split/test/
            idle/
            left/
            right/

    Each CSV file should contain:

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

        # Normalize MPU6050-style raw accelerometer values.
        data = data / self.normalize_scale

        # Flatten 64 x 3 into 192.
        data = data.flatten()

        x = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


# ----------------------------------------------
# BitNetMCU training
# ----------------------------------------------

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
        for start in range(0, len(all_test_images), batch_size):
            images = all_test_images[start:start + batch_size]
            model(images)

    for hook in hooks:
        hook.remove()

    if total_activations == 0:
        fraction_positive = 0.0
    else:
        fraction_positive = positive_activations / total_activations

    writer.add_scalar("Activations/positive_fraction", fraction_positive, epoch + 1)

    return fraction_positive


def add_mask_regularization(model, lambda_l1):
    mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

    if mask_layer is None:
        return 0

    l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
    return l1_reg


def train_model(model, device, hyperparameters, train_data, test_data):
    num_epochs = hyperparameters["num_epochs"]
    learning_rate = hyperparameters["learning_rate"]
    halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
    runname = create_run_name(hyperparameters)

    batch_size = hyperparameters["batch_size"]

    if hyperparameters["augmentation"]:
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=len(train_data),
            shuffle=False,
        )

        entire_dataset = next(iter(train_loader))
        all_train_images = entire_dataset[0].to(device)
        all_train_labels = entire_dataset[1].to(device)

    test_loader = DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
    )

    entire_dataset = next(iter(test_loader))
    all_test_images = entire_dataset[0].to(device)
    all_test_labels = entire_dataset[1].to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if hyperparameters["scheduler"] == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=hyperparameters["step_size"],
            gamma=hyperparameters["lr_decay"],
        )
    elif hyperparameters["scheduler"] == "Cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=0,
        )
    elif hyperparameters["scheduler"] == "CosineWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=hyperparameters["T_0"],
            T_mult=hyperparameters["T_mult"],
            eta_min=0,
        )
    else:
        raise ValueError("Invalid scheduler")

    criterion = nn.CrossEntropyLoss()

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{runname}-{now_str}")

    trainaccuracy = 0.0
    testaccuracy = 0.0
    totalbits = 0
    train_loss = [0.0]
    test_loss = [0.0]

    for epoch in range(num_epochs):
        all_preds = []
        all_labels_list = []
        all_probs = []

        correct = 0
        train_loss = []
        start_time = time.time()

        model.train()

        if hyperparameters["augmentation"]:
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if epoch < hyperparameters["prune_epoch"]:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()

        else:
            indices = list(range(len(all_train_images)))
            random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]

                images = torch.stack([all_train_images[i] for i in batch_indices])
                labels = torch.stack([all_train_labels[i] for i in batch_indices])

                optimizer.zero_grad()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                if epoch < hyperparameters["prune_epoch"]:
                    loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())
                correct += (predicted == labels).sum().item()

        scheduler.step()

        if epoch + 1 == halve_lr_epoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5

            print(f"Learning rate halved at epoch {epoch + 1}")

        trainaccuracy = correct / len(train_data) * 100.0

        model.eval()

        correct = 0
        total = 0
        test_loss = []

        with torch.no_grad():
            for start in range(0, len(all_test_images), batch_size):
                images = all_test_images[start:start + batch_size]
                labels = all_test_labels[start:start + batch_size]

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                loss = criterion(outputs, labels)
                test_loss.append(loss.item())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        activity = log_positive_activations(
            model,
            writer,
            epoch,
            all_test_images,
            batch_size,
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        testaccuracy = correct / total * 100.0

        cm = confusion_matrix(all_labels_list, all_preds)
        f1 = f1_score(all_labels_list, all_preds, average="macro")

        print("Confusion Matrix:\n", cm)
        print("F1 Score:", f1)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"LTrain:{np.mean(train_loss):.6f} "
            f"ATrain:{trainaccuracy:.2f}% "
            f"LTest:{np.mean(test_loss):.6f} "
            f"ATest:{testaccuracy:.2f}% "
            f"Time[s]:{epoch_time:.2f} "
            f"Act:{activity * 100:.1f}% "
            f"w_clip/entropy[bits]: ",
            end="",
        )

        print("\nClassification Report:\n")
        print(classification_report(all_labels_list, all_preds))

        try:
            roc_auc = roc_auc_score(all_labels_list, all_probs, multi_class="ovr")
            print("ROC AUC:", roc_auc)
        except Exception:
            print("ROC not available")

        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)

        sensitivity = TP / np.maximum(TP + FN, 1)
        specificity = TN / np.maximum(TN + FP, 1)

        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

        totalbits = 0

        for layer in model.modules():
            if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):
                if epoch < hyperparameters["maxw_update_until_epoch"]:
                    layer.update_clipping_scalar(
                        layer.weight,
                        hyperparameters["maxw_algo"],
                        hyperparameters["maxw_quantscale"],
                    )

                w_quant, _, _ = layer.weight_quant(layer.weight)
                _, counts = np.unique(
                    w_quant.cpu().detach().numpy(),
                    return_counts=True,
                )

                probabilities = counts / np.sum(counts)
                entropy = -np.sum(probabilities * np.log2(probabilities))

                print(f"{layer.s.item():.3f}/{entropy:.2f}", end=" ")

                totalbits += layer.weight.numel() * layer.bpw

        print()

        if epoch + 1 == hyperparameters["prune_epoch"]:
            for m in model.modules():
                if isinstance(m, MaskingLayer):
                    m.prune_channels(
                        prune_number=hyperparameters["prune_groupstoprune"],
                        groups=hyperparameters["prune_totalgroups"],
                    )

        writer.add_scalar("Loss/train", np.mean(train_loss), epoch + 1)
        writer.add_scalar("Accuracy/train", trainaccuracy, epoch + 1)
        writer.add_scalar("Loss/test", np.mean(test_loss), epoch + 1)
        writer.add_scalar("Accuracy/test", testaccuracy, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
        writer.flush()

    numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"TotalBits: {totalbits} TotalBytes: {totalbits / 8.0}")

    writer.add_hparams(
        hyperparameters,
        {
            "Parameters": numofweights,
            "Totalbits": totalbits,
            "Accuracy/train": trainaccuracy,
            "Accuracy/test": testaccuracy,
            "Loss/train": np.mean(train_loss),
            "Loss/test": np.mean(test_loss),
        },
    )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--params",
        type=str,
        help="Name of the parameter file",
        default="trainingparameters.yaml",
    )

    args = parser.parse_args()

    paramname = args.params if args.params else "trainingparameters.yaml"

    print(f"Load parameters from file: {paramname}")

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    runname = create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    train_data = None
    test_data = None

    # ------------------------------------------
    # SIGNAL dataset path
    # ------------------------------------------

    if dataset_name == "SIGNAL":
        signal_classes = hyperparameters.get(
            "signal_classes",
            ["idle", "left", "right"],
        )

        signal_train_dir = hyperparameters.get(
            "signal_train_dir",
            "signal_dataset/split/train",
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

        train_data = SignalDataset(
            root_dir=signal_train_dir,
            class_names=signal_classes,
            samples_per_window=samples_per_window,
            channels=signal_channels,
            normalize_scale=normalize_scale,
        )

        test_data = SignalDataset(
            root_dir=signal_test_dir,
            class_names=signal_classes,
            samples_per_window=samples_per_window,
            channels=signal_channels,
            normalize_scale=normalize_scale,
        )

        hyperparameters["num_classes"] = num_classes
        hyperparameters["input_size"] = input_size

        print("Loaded SIGNAL dataset")
        print("Classes:", signal_classes)
        print("Train samples:", len(train_data))
        print("Test samples:", len(test_data))
        print("Input size:", input_size)

    # ------------------------------------------
    # Image dataset paths
    # ------------------------------------------

    else:
        if dataset_name == "MNIST":
            num_classes = 10
            mean, std = (0.1307,), (0.3081,)
            base_dataset_train = datasets.MNIST
            base_dataset_test = datasets.MNIST
            dataset_kwargs = {"train": True}
            dataset_kwargs_test = {"train": False}

        elif dataset_name.startswith("EMNIST"):
            split = dataset_name.split("_")[1].lower() if "_" in dataset_name else "balanced"

            split_alias = {
                "BALANCED": "balanced",
                "BYCLASS": "byclass",
                "BYMERGE": "bymerge",
                "LETTERS": "letters",
                "DIGITS": "digits",
                "MNIST": "mnist",
            }

            split = split_alias.get(split.upper(), split)

            split_classes = {
                "byclass": 62,
                "bymerge": 47,
                "balanced": 47,
                "letters": 37,
                "digits": 10,
                "mnist": 10,
            }

            num_classes = split_classes.get(split, 47)
            mean, std = (0.1307,), (0.3081,)

            from torchvision.datasets import EMNIST

            base_dataset_train = EMNIST
            base_dataset_test = EMNIST
            dataset_kwargs = {"split": split, "train": True}
            dataset_kwargs_test = {"split": split, "train": False}

        elif dataset_name == "FASHION":
            num_classes = 10
            mean, std = (0.5,), (0.5,)
            base_dataset_train = datasets.FashionMNIST
            base_dataset_test = datasets.FashionMNIST
            dataset_kwargs = {"train": True}
            dataset_kwargs_test = {"train": False}

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        transform = transforms.Compose(
            [
                transforms.Resize((16, 16)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        train_data = base_dataset_train(
            root="data",
            transform=transform,
            download=True,
            **dataset_kwargs,
        )

        test_data = base_dataset_test(
            root="data",
            transform=transform,
            download=True,
            **dataset_kwargs_test,
        )

        if hyperparameters["augmentation"]:
            augmented_transform = transforms.Compose(
                [
                    transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
                    transforms.RandomAffine(
                        degrees=hyperparameters["rotation2"],
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1),
                    ),
                    transforms.RandomApply(
                        [
                            transforms.ElasticTransform(alpha=40.0, sigma=4.0),
                        ],
                        p=hyperparameters["elastictransformprobability"],
                    ),
                    transforms.Resize((16, 16)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

            augmented_train_data = base_dataset_train(
                root="data",
                transform=augmented_transform,
                download=True,
                **dataset_kwargs,
            )

            train_data = ConcatDataset([train_data, augmented_train_data])

        hyperparameters["num_classes"] = num_classes

    # ------------------------------------------
    # Model
    # ------------------------------------------

    model = load_model(
        hyperparameters["model"],
        {**hyperparameters, "num_classes": hyperparameters["num_classes"]},
    )

    model = model.to(device)

    if dataset_name == "SIGNAL":
        summary(model, input_size=(hyperparameters["input_size"],))
    else:
        summary(model, input_size=(1, 16, 16))

    print("training...")
    train_model(model, device, hyperparameters, train_data, test_data)

    print("saving model...")

    Path("modeldata").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        f"modeldata/{runname}.pth",
    )
