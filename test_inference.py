# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import numpy as np
# from BitNetMCU import QuantizedModel
# # from models import FCMNIST
# from ctypes import CDLL, c_uint32, c_int8, c_uint8, POINTER
# import argparse
# import yaml
# import importlib

# # Export quantized model from saved checkpoint
# # cpldcpu 2024-04-14
# # Note: Hyperparameters are used to generated the filename
# #---------------------------------------------

# def create_run_name(hyperparameters):
#     runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
#     hyperparameters["runname"] = runname
#     return runname

# def load_model(model_name, params):
#     try:
#         module = importlib.import_module('models')
#         model_class = getattr(module, model_name)
#         kwargs = dict(
#             network_width1=params["network_width1"],
#             network_width2=params["network_width2"],
#             network_width3=params["network_width3"],
#             QuantType=params["QuantType"],
#             NormType=params["NormType"],
#             WScale=params["WScale"]
#         )
#         if 'cnn_width' in params:
#             kwargs['cnn_width'] = params['cnn_width']
#         return model_class(**kwargs)
#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py")
    
# def export_test_data_to_c(test_loader, filename, num=8):
#     with open(filename, 'w') as f:
#         for i, (input_data, labels) in enumerate(test_loader):
#             if i >= num:
#                 break
#             # Reshape and convert to numpy
#             input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#             labels = labels.cpu().numpy()

#             scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#             scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.uint8)

#             f.write(f'int8_t input_data_{i}[256] = {{\n')
#             flattened_data = scaled_data.flatten()
#             for k in range(0, len(flattened_data), 16):
#                 f.write(', '.join(f'0x{value:02X}' for value in flattened_data[k:k+16]) + ',\n')
#             f.write('};\n')

#             f.write(f'uint8_t label_{i} = ' + str(labels[0]) + ';\n')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Training script')
#     parser.add_argument('--params', type=str, help='Name of the parameter file', default='trainingparameters.yaml')
    
#     args = parser.parse_args()
    
#     if args.params:
#         paramname = args.params
#     else:
#         paramname = 'trainingparameters.yaml'

#     print(f'Load parameters from file: {paramname}')
#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     # main
#     runname= create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the MNIST dataset

#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     if dataset_name == "MNIST":
#         mean, std = (0.1307,), (0.3081,)
#         dataset_cls = datasets.MNIST
#     elif dataset_name == "FASHION":
#         mean, std = (0.2860,), (0.3530,)
#         dataset_cls = datasets.FashionMNIST
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     transform = transforms.Compose([
#         transforms.Resize((16, 16)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])

#     train_data = dataset_cls(root='data', train=True, transform=transform, download=True)
#     test_data = dataset_cls(root='data', train=False, transform=transform, download=True)
    
#     # Create data loaders
#     test_loader = DataLoader(test_data, batch_size=hyperparameters["batch_size"], shuffle=False)

#     model = load_model(hyperparameters["model"], hyperparameters).to(device)
    
#     print('Loading model...')    
#     try:
#         # model.load_state_dict(torch.load(f'modeldata/{runname}.pth'))
#         model.load_state_dict(torch.load(f'modeldata/{runname}.pth', map_location=torch.device('cpu')))
#     except FileNotFoundError:
#         print(f"The file 'modeldata/{runname}.pth' does not exist.")
#         exit()

#     print('Inference using the original model...')
#     correct = 0
#     total = 0
#     test_loss = []
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)        
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     testaccuracy = correct / total * 100
#     print(f'Accuracy/Test of trained model: {testaccuracy} %')

#     print('Quantizing model...')
#     # Quantize the model
#     quantized_model = QuantizedModel(model)
#     print(f'Total number of bits: {quantized_model.totalbits()} ({quantized_model.totalbits()/8/1024} kbytes)')

#     # Inference using the quantized model
#     print ("Verifying inference of quantized model in Python and C")

#    # Initialize counter
#     counter = 0
#     correct_c = 0
#     correct_py = 0
#     mismatch = 0

#     test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)    

#     # export_test_data_to_c(test_loader2, 'BitNetMCU_MNIST_test_data.h', num=10)

#     lib = CDLL('./Bitnet_inf.dll')

#     for input_data, labels in test_loader2:
#         input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#         labels = labels.cpu().numpy()

#         scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#         scaled_data = np.round(input_data * scale).clip(-128, 127) 

#         # Create a pointer to the ctypes array
#         input_data_pointer = (c_int8 * len(scaled_data.flatten()))(*scaled_data.astype(np.int8).flatten())

#         lib.Inference.argtypes = [POINTER(c_int8)]
#         lib.Inference.restype = c_uint32

#         # Inference C
#         result_c = lib.Inference(input_data_pointer)

#         # Inference Python
#         result_py = quantized_model.inference_quantized(input_data)
#         predict_py = np.argmax(result_py, axis=1)

#         # activations = quantized_model.get_activations(input_data)

#         if (result_c == labels[0]):
#             correct_c += 1

#         if (predict_py[0] == labels[0]):
#             correct_py += 1

#         if (result_c != predict_py[0]):
#             print(f'{counter:5} Mismatch between inference engines found. Prediction C: {result_c} Prediction Python: {predict_py[0]} True: {labels[0]}')
#             mismatch +=1

#         counter += 1

#     print("size of test data:", counter)
#     print(f'Mispredictions C: {counter - correct_c} Py: {counter - correct_py}')
#     print('Overall accuracy C:', correct_c / counter * 100, '%')
#     print('Overall accuracy Python:', correct_py / counter * 100, '%')
    
#     print(f'Mismatches between engines: {mismatch} ({mismatch/counter*100}%)')

# import torch
# from torchvision import datasets, transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# import numpy as np
# from BitNetMCU import QuantizedModel
# # from models import FCMNIST
# from ctypes import CDLL, c_uint32, c_int8, c_uint8, POINTER
# import argparse
# import yaml
# import importlib
# import os

# # Export quantized model from saved checkpoint
# # cpldcpu 2024-04-14
# # Note: Hyperparameters are used to generated the filename
# #---------------------------------------------

# def create_run_name(hyperparameters):
#     runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
#     hyperparameters["runname"] = runname
#     return runname


# def load_model(model_name, params):
#     try:
#         module = importlib.import_module('models')
#         model_class = getattr(module, model_name)

#         kwargs = dict(
#             network_width1=params["network_width1"],
#             network_width2=params["network_width2"],
#             network_width3=params["network_width3"],
#             QuantType=params["QuantType"],
#             NormType=params["NormType"],
#             WScale=params["WScale"]
#         )

#         if 'cnn_width' in params:
#             kwargs['cnn_width'] = params['cnn_width']

#         # Important for Olivetti: 40 output classes
#         if 'num_classes' in params:
#             kwargs['num_classes'] = params['num_classes']

#         return model_class(**kwargs)

#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py")


# def export_test_data_to_c(test_loader, filename, num=8):
#     with open(filename, 'w') as f:
#         for i, (input_data, labels) in enumerate(test_loader):
#             if i >= num:
#                 break

#             # Reshape and convert to numpy
#             input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#             labels = labels.cpu().numpy()

#             scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#             scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.uint8)

#             f.write(f'int8_t input_data_{i}[256] = {{\n')

#             flattened_data = scaled_data.flatten()

#             for k in range(0, len(flattened_data), 16):
#                 f.write(', '.join(f'0x{value:02X}' for value in flattened_data[k:k+16]) + ',\n')

#             f.write('};\n')

#             f.write(f'uint8_t label_{i} = ' + str(labels[0]) + ';\n')


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Inference script')
#     parser.add_argument('--params', type=str, help='Name of the parameter file', default='trainingparameters.yaml')

#     args = parser.parse_args()

#     if args.params:
#         paramname = args.params
#     else:
#         paramname = 'trainingparameters.yaml'

#     print(f'Load parameters from file: {paramname}')

#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     # main
#     runname = create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load dataset
#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     if dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         dataset_cls = datasets.MNIST

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = dataset_cls(root='data', train=True, transform=transform, download=True)
#         test_data = dataset_cls(root='data', train=False, transform=transform, download=True)

#     elif dataset_name == "FASHION":
#         num_classes = 10
#         mean, std = (0.2860,), (0.3530,)
#         dataset_cls = datasets.FashionMNIST

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = dataset_cls(root='data', train=True, transform=transform, download=True)
#         test_data = dataset_cls(root='data', train=False, transform=transform, download=True)

#     elif dataset_name == "OLIVETTI":
#         # ============================================================
#         # Olivetti Face Dataset
#         #
#         # Same original structure:
#         # train_data = training set
#         # test_data  = final testing set
#         #
#         # This evaluates final test accuracy.
#         # It does NOT use validation_set here.
#         # ============================================================

#         num_classes = 40
#         mean, std = (0.5,), (0.5,)

#         data_root = hyperparameters["data_root"]

#         train_folder = hyperparameters.get(
#             "train_folder",
#             "olivetti_training_set_ONLY"
#         )

#         test_folder = hyperparameters.get(
#             "test_folder",
#             "olivetti_FINAL_TEST_SET_LOCKED_ONLY"
#         )

#         train_dir = os.path.join(data_root, train_folder)
#         test_dir = os.path.join(data_root, test_folder)

#         print("Olivetti train folder:", train_dir)
#         print("Olivetti TEST folder:", test_dir)

#         if not os.path.exists(train_dir):
#             raise FileNotFoundError(f"Training folder not found: {train_dir}")

#         if not os.path.exists(test_dir):
#             raise FileNotFoundError(f"Testing folder not found: {test_dir}")

#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = ImageFolder(root=train_dir, transform=transform)
#         test_data = ImageFolder(root=test_dir, transform=transform)

#         if train_data.class_to_idx != test_data.class_to_idx:
#              raise ValueError(
#                  "Class mapping mismatch between training and test folders. "
#                  "Please make sure both folders contain the same class_00 to class_39 structure."
#              )
            
#         if len(train_data.classes) != 40:
#             raise ValueError(
#                 f"Olivetti training folder should contain 40 classes, "
#                 f"but found {len(train_data.classes)} classes."
#             )

#         if len(test_data.classes) != 40:
#             raise ValueError(
#                 f"Olivetti test folder should contain 40 classes, "
#                 f"but found {len(test_data.classes)} classes."
#             )
            
#         print("Olivetti class mapping:", test_data.class_to_idx)
#         print("Training samples:", len(train_data))
#         print("Testing samples:", len(test_data))

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     # Important: pass number of classes to model
#     hyperparameters["num_classes"] = num_classes

#     # Create data loaders
#     test_loader = DataLoader(
#         test_data,
#         batch_size=hyperparameters["batch_size"],
#         shuffle=False
#     )

#     model = load_model(hyperparameters["model"], hyperparameters).to(device)

#     # Check model output
#     dummy_input = torch.randn(1, 1, 16, 16).to(device)

#     with torch.no_grad():
#         dummy_output = model(dummy_input)

#     print("Model output shape:", dummy_output.shape)
#     print("Expected classes:", num_classes)

#     if dummy_output.shape[1] != num_classes:
#         raise ValueError(
#             f"Model output mismatch. "
#             f"Model outputs {dummy_output.shape[1]} classes, "
#             f"but dataset needs {num_classes} classes. "
#             f"Please modify models.py final layer to use num_classes."
#         )

#     print('Loading model...')

#     try:
#         model.load_state_dict(
#             torch.load(
#                 f'modeldata/{runname}.pth',
#                 map_location=torch.device('cpu')
#             )
#         )

#     except FileNotFoundError:
#         print(f"The file 'modeldata/{runname}.pth' does not exist.")
#         exit()

#     model = model.to(device)
#     model.eval()

#     print('Inference using the original model...')

#     correct = 0
#     total = 0
#     test_loss = []

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     testaccuracy = correct / total * 100

#     print(f'Accuracy/Test of trained model: {testaccuracy} %')

#     print('Quantizing model...')

#     # Quantize the model
#     quantized_model = QuantizedModel(model)

#     print(f'Total number of bits: {quantized_model.totalbits()} ({quantized_model.totalbits()/8/1024} kbytes)')

#     # Inference using the quantized model
#     print("Verifying inference of quantized model in Python and C")

#     # Initialize counter
#     counter = 0
#     correct_c = 0
#     correct_py = 0
#     mismatch = 0

#     test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

#     # Optional: export some final test samples to C header
#     # export_test_data_to_c(test_loader2, 'BitNetMCU_Olivetti_test_data.h', num=10)

#     dll_path = hyperparameters.get("dll_path", "./Bitnet_inf.dll")

#     if not os.path.exists(dll_path):
#         raise FileNotFoundError(
#             f"Cannot find DLL file: {dll_path}\n"
#             "Please make sure Bitnet_inf.dll is compiled from the exported Olivetti model."
#         )

#     lib = CDLL(dll_path)

#     lib.Inference.argtypes = [POINTER(c_int8)]
#     lib.Inference.restype = c_uint32

#     for input_data, labels in test_loader2:
#         input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#         labels = labels.cpu().numpy()

#         scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#         scaled_data = np.round(input_data * scale).clip(-128, 127)

#         # Create a pointer to the ctypes array
#         input_data_pointer = (c_int8 * len(scaled_data.flatten()))(*scaled_data.astype(np.int8).flatten())

#         # Inference C
#         result_c = lib.Inference(input_data_pointer)

#         # Inference Python
#         result_py = quantized_model.inference_quantized(input_data)
#         predict_py = np.argmax(result_py, axis=1)

#         if result_c == labels[0]:
#             correct_c += 1

#         if predict_py[0] == labels[0]:
#             correct_py += 1

#         if result_c != predict_py[0]:
#             print(
#                 f'{counter:5} Mismatch between inference engines found. '
#                 f'Prediction C: {result_c} '
#                 f'Prediction Python: {predict_py[0]} '
#                 f'True: {labels[0]}'
#             )
#             mismatch += 1

#         counter += 1

#     print("size of test data:", counter)
#     print(f'Mispredictions C: {counter - correct_c} Py: {counter - correct_py}')
#     print('Overall accuracy C:', correct_c / counter * 100, '%')
#     print('Overall accuracy Python:', correct_py / counter * 100, '%')

#     print(f'Mismatches between engines: {mismatch} ({mismatch/counter*100}%)')

# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from BitNetMCU import QuantizedModel, BitLinear, Activation
# from ctypes import CDLL, c_uint32, c_int8, POINTER
# import argparse
# import yaml
# import importlib
# import os
# import pandas as pd
# import pickle
# import json


# # ----------------------------------------------------------
# # Run name
# # ----------------------------------------------------------

# def create_run_name(hyperparameters):
#     runname = (
#         hyperparameters["runtag"]
#         + "_"
#         + hyperparameters["model"]
#         + ("_Aug" if hyperparameters.get("augmentation", False) else "")
#         + "_BitMnist_"
#         + hyperparameters["QuantType"]
#         + "_width"
#         + str(hyperparameters["network_width1"])
#         + "_"
#         + str(hyperparameters["network_width2"])
#         + "_"
#         + str(hyperparameters["network_width3"])
#         + "_epochs"
#         + str(hyperparameters["num_epochs"])
#     )
#     hyperparameters["runname"] = runname
#     return runname


# # ----------------------------------------------------------
# # Gate Driver MLP
# # Must match training.py
# # ----------------------------------------------------------

# def make_bitlinear(in_features, out_features, QuantType, NormType, WScale):
#     try:
#         return BitLinear(
#             in_features,
#             out_features,
#             QuantType=QuantType,
#             NormType=NormType,
#             WScale=WScale
#         )
#     except TypeError:
#         try:
#             return BitLinear(
#                 in_features,
#                 out_features,
#                 QuantType,
#                 NormType,
#                 WScale
#             )
#         except TypeError:
#             print("Warning: BitLinear failed. Falling back to nn.Linear.")
#             return nn.Linear(in_features, out_features)


# class GateDriverMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         network_width1=64,
#         network_width2=32,
#         network_width3=0,
#         QuantType="4bitsym",
#         NormType="RMS",
#         WScale="PerTensor",
#         num_classes=4,
#         dropout=0.1,
#         **kwargs
#     ):
#         super().__init__()

#         layers = []

#         layers.append(make_bitlinear(input_dim, network_width1, QuantType, NormType, WScale))
#         layers.append(Activation())

#         layers.append(make_bitlinear(network_width1, network_width2, QuantType, NormType, WScale))
#         layers.append(Activation())

#         if network_width3 and network_width3 > 0:
#             layers.append(make_bitlinear(network_width2, network_width3, QuantType, NormType, WScale))
#             layers.append(Activation())
#             layers.append(make_bitlinear(network_width3, num_classes, QuantType, NormType, WScale))
#         else:
#             layers.append(make_bitlinear(network_width2, num_classes, QuantType, NormType, WScale))

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


# # ----------------------------------------------------------
# # Load model
# # ----------------------------------------------------------

# def load_model(model_name, params):
#     if model_name == "GateDriverMLP":
#         return GateDriverMLP(**params)

#     try:
#         module = importlib.import_module("models")
#         model_class = getattr(module, model_name)

#         kwargs = dict(
#             network_width1=params["network_width1"],
#             network_width2=params["network_width2"],
#             network_width3=params["network_width3"],
#             QuantType=params["QuantType"],
#             NormType=params["NormType"],
#             WScale=params["WScale"]
#         )

#         if "cnn_width" in params:
#             kwargs["cnn_width"] = params["cnn_width"]

#         if "num_classes" in params:
#             kwargs["num_classes"] = params["num_classes"]

#         return model_class(**kwargs)

#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py or test_inference.py")


# # ----------------------------------------------------------
# # Gate Driver Excel Test Loader
# # ----------------------------------------------------------

# def load_gate_driver_test_excel(test_file, label_col="label"):
#     if not os.path.exists(test_file):
#         raise FileNotFoundError(f"Test Excel file not found: {test_file}")

#     test_df = pd.read_excel(test_file)

#     if label_col not in test_df.columns:
#         raise ValueError(f"Label column '{label_col}' not found in test file.")

#     feature_cols_path = "modeldata/gate_driver_feature_cols.json"
#     label_mapping_path = "modeldata/gate_driver_label_mapping.json"
#     scaler_path = "modeldata/gate_driver_scaler.pkl"

#     if not os.path.exists(feature_cols_path):
#         raise FileNotFoundError("Missing modeldata/gate_driver_feature_cols.json. Run training.py first.")

#     if not os.path.exists(label_mapping_path):
#         raise FileNotFoundError("Missing modeldata/gate_driver_label_mapping.json. Run training.py first.")

#     if not os.path.exists(scaler_path):
#         raise FileNotFoundError("Missing modeldata/gate_driver_scaler.pkl. Run training.py first.")

#     with open(feature_cols_path, "r") as f:
#         feature_cols = json.load(f)

#     with open(label_mapping_path, "r") as f:
#         label_to_id = json.load(f)

#     with open(scaler_path, "rb") as f:
#         scaler = pickle.load(f)

#     missing_cols = [c for c in feature_cols if c not in test_df.columns]
#     if len(missing_cols) > 0:
#         raise ValueError(f"Test file missing required feature columns: {missing_cols}")

#     X_test = test_df[feature_cols].copy()
#     X_test = X_test.fillna(X_test.median(numeric_only=True))
#     X_test = scaler.transform(X_test).astype("float32")

#     y_raw = test_df[label_col].astype(str)

#     unknown_labels = set(y_raw.unique()) - set(label_to_id.keys())
#     if len(unknown_labels) > 0:
#         raise ValueError(f"Test file has labels not seen during training: {unknown_labels}")

#     y_test = y_raw.map(label_to_id).values.astype("int64")

#     test_data = TensorDataset(
#         torch.tensor(X_test, dtype=torch.float32),
#         torch.tensor(y_test, dtype=torch.long)
#     )

#     input_dim = len(feature_cols)
#     num_classes = len(label_to_id)

#     print("Gate-driver TEST Excel:", test_file)
#     print("Test samples:", len(test_data))
#     print("Input dimension:", input_dim)
#     print("Number of classes:", num_classes)
#     print("Label mapping:", label_to_id)

#     return test_data, num_classes, input_dim


# # ----------------------------------------------------------
# # Export image test data only
# # ----------------------------------------------------------

# def export_test_data_to_c(test_loader, filename, num=8):
#     with open(filename, "w") as f:
#         for i, (input_data, labels) in enumerate(test_loader):
#             if i >= num:
#                 break

#             input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#             labels = labels.cpu().numpy()

#             scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#             scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.uint8)

#             f.write(f"int8_t input_data_{i}[256] = {{\n")

#             flattened_data = scaled_data.flatten()

#             for k in range(0, len(flattened_data), 16):
#                 f.write(", ".join(f"0x{value:02X}" for value in flattened_data[k:k + 16]) + ",\n")

#             f.write("};\n")
#             f.write(f"uint8_t label_{i} = {labels[0]};\n")


# # ----------------------------------------------------------
# # Main
# # ----------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Inference script")
#     parser.add_argument(
#         "--params",
#         type=str,
#         help="Name of the parameter file",
#         default="trainingparameters.yaml"
#     )

#     args = parser.parse_args()
#     paramname = args.params

#     print(f"Load parameters from file: {paramname}")

#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     hyperparameters.setdefault("augmentation", False)
#     hyperparameters.setdefault("network_width3", 0)
#     hyperparameters.setdefault("dropout", 0.1)

#     runname = create_run_name(hyperparameters)
#     print("Run name:", runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     input_dim = None

#     # ----------------------------------------------------------
#     # GATE DRIVER EXCEL TEST
#     # ----------------------------------------------------------

#     if dataset_name == "GATE_DRIVER":
#         test_data, num_classes, input_dim = load_gate_driver_test_excel(
#             test_file=hyperparameters["test_file"],
#             label_col=hyperparameters.get("label_col", "label")
#         )

#         hyperparameters["input_dim"] = input_dim

#     # ----------------------------------------------------------
#     # MNIST
#     # ----------------------------------------------------------

#     elif dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         dataset_cls = datasets.MNIST

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         test_data = dataset_cls(
#             root="data",
#             train=False,
#             transform=transform,
#             download=True
#         )

#     # ----------------------------------------------------------
#     # FASHION MNIST
#     # ----------------------------------------------------------

#     elif dataset_name == "FASHION":
#         num_classes = 10
#         mean, std = (0.2860,), (0.3530,)
#         dataset_cls = datasets.FashionMNIST

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         test_data = dataset_cls(
#             root="data",
#             train=False,
#             transform=transform,
#             download=True
#         )

#     # ----------------------------------------------------------
#     # OLIVETTI FINAL TEST
#     # ----------------------------------------------------------

#     elif dataset_name == "OLIVETTI":
#         num_classes = 40
#         mean, std = (0.5,), (0.5,)

#         data_root = hyperparameters["data_root"]

#         train_folder = hyperparameters.get(
#             "train_folder",
#             "olivetti_training_set_ONLY"
#         )

#         test_folder = hyperparameters.get(
#             "test_folder",
#             "olivetti_FINAL_TEST_SET_LOCKED_ONLY"
#         )

#         train_dir = os.path.join(data_root, train_folder)
#         test_dir = os.path.join(data_root, test_folder)

#         print("Olivetti train folder:", train_dir)
#         print("Olivetti TEST folder:", test_dir)

#         if not os.path.exists(train_dir):
#             raise FileNotFoundError(f"Training folder not found: {train_dir}")

#         if not os.path.exists(test_dir):
#             raise FileNotFoundError(f"Testing folder not found: {test_dir}")

#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = ImageFolder(root=train_dir, transform=transform)
#         test_data = ImageFolder(root=test_dir, transform=transform)

#         if train_data.class_to_idx != test_data.class_to_idx:
#             raise ValueError(
#                 "Class mapping mismatch between training and test folders. "
#                 "Please make sure both folders contain the same class structure."
#             )

#         print("Olivetti class mapping:", test_data.class_to_idx)
#         print("Testing samples:", len(test_data))

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     hyperparameters["num_classes"] = num_classes

#     test_loader = DataLoader(
#         test_data,
#         batch_size=hyperparameters["batch_size"],
#         shuffle=False
#     )

#     model = load_model(hyperparameters["model"], hyperparameters).to(device)

#     # ----------------------------------------------------------
#     # Check model output
#     # ----------------------------------------------------------

#     if dataset_name == "GATE_DRIVER":
#         dummy_input = torch.randn(1, input_dim).to(device)
#     else:
#         dummy_input = torch.randn(1, 1, 16, 16).to(device)

#     with torch.no_grad():
#         dummy_output = model(dummy_input)

#     print("Model output shape:", dummy_output.shape)
#     print("Expected classes:", num_classes)

#     if dummy_output.shape[1] != num_classes:
#         raise ValueError(
#             f"Model output mismatch. "
#             f"Model outputs {dummy_output.shape[1]} classes, "
#             f"but dataset needs {num_classes} classes."
#         )

#     # ----------------------------------------------------------
#     # Load trained checkpoint
#     # ----------------------------------------------------------

#     print("Loading model...")

#     model_path = f"modeldata/{runname}.pth"
#     best_model_path = f"modeldata/{runname}_best.pth"

#     if os.path.exists(best_model_path):
#         model_path = best_model_path

#     if not os.path.exists(model_path):
#         raise FileNotFoundError(
#             f"Cannot find model checkpoint:\n"
#             f"{model_path}\n"
#             f"Expected either:\n"
#             f"modeldata/{runname}.pth\n"
#             f"or\n"
#             f"modeldata/{runname}_best.pth"
#         )

#     model.load_state_dict(
#         torch.load(
#             model_path,
#             map_location=torch.device("cpu")
#         )
#     )

#     print("Loaded checkpoint:", model_path)

#     model = model.to(device)
#     model.eval()

#     # ----------------------------------------------------------
#     # Original model inference
#     # ----------------------------------------------------------

#     print("Inference using original trained model...")

#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)

#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     testaccuracy = correct / total * 100
#     print(f"Accuracy/Test of trained model: {testaccuracy:.2f} %")

#     # ----------------------------------------------------------
#     # Quantized Python model inference
#     # ----------------------------------------------------------

#     print("Quantizing model...")

#     quantized_model = QuantizedModel(model)

#     total_bits = quantized_model.totalbits()
#     print(f"Total number of bits: {total_bits} ({total_bits / 8 / 1024:.4f} kbytes)")

#     print("Inference using quantized Python model...")

#     test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

#     counter = 0
#     correct_py = 0

#     for input_data, labels in test_loader2:
#         input_data_np = input_data.view(input_data.size(0), -1).cpu().numpy()
#         labels_np = labels.cpu().numpy()

#         result_py = quantized_model.inference_quantized(input_data_np)
#         predict_py = np.argmax(result_py, axis=1)

#         if predict_py[0] == labels_np[0]:
#             correct_py += 1

#         counter += 1

#     print("Size of test data:", counter)
#     print("Mispredictions Python:", counter - correct_py)
#     print("Overall accuracy Python:", correct_py / counter * 100, "%")

#     # ----------------------------------------------------------
#     # C DLL inference
#     # Only run for image datasets first.
#     # Gate-driver Excel input dimension is not fixed 256.
#     # ----------------------------------------------------------

#     if dataset_name == "GATE_DRIVER":
#         print("C DLL inference skipped for GATE_DRIVER.")
#         print("Reason: Excel tabular input dimension is not fixed 16x16 image input.")
#         print("Python quantized inference completed successfully.")

#     else:
#         print("Verifying inference of quantized model in Python and C...")

#         dll_path = hyperparameters.get("dll_path", "./Bitnet_inf.dll")

#         if not os.path.exists(dll_path):
#             raise FileNotFoundError(
#                 f"Cannot find DLL file: {dll_path}\n"
#                 "Please make sure Bitnet_inf.dll is compiled from the exported model."
#             )

#         lib = CDLL(dll_path)

#         lib.Inference.argtypes = [POINTER(c_int8)]
#         lib.Inference.restype = c_uint32

#         counter = 0
#         correct_c = 0
#         correct_py = 0
#         mismatch = 0

#         for input_data, labels in test_loader2:
#             input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
#             labels = labels.cpu().numpy()

#             scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
#             scaled_data = np.round(input_data * scale).clip(-128, 127)

#             input_data_pointer = (
#                 c_int8 * len(scaled_data.flatten())
#             )(*scaled_data.astype(np.int8).flatten())

#             result_c = lib.Inference(input_data_pointer)

#             result_py = quantized_model.inference_quantized(input_data)
#             predict_py = np.argmax(result_py, axis=1)

#             if result_c == labels[0]:
#                 correct_c += 1

#             if predict_py[0] == labels[0]:
#                 correct_py += 1

#             if result_c != predict_py[0]:
#                 print(
#                     f"{counter:5} Mismatch between inference engines found. "
#                     f"Prediction C: {result_c} "
#                     f"Prediction Python: {predict_py[0]} "
#                     f"True: {labels[0]}"
#                 )
#                 mismatch += 1

#             counter += 1

#         print("Size of test data:", counter)
#         print(f"Mispredictions C: {counter - correct_c} Py: {counter - correct_py}")
#         print("Overall accuracy C:", correct_c / counter * 100, "%")
#         print("Overall accuracy Python:", correct_py / counter * 100, "%")
#         print(f"Mismatches between engines: {mismatch} ({mismatch / counter * 100}%)")

import os
import json
import pickle
import argparse

import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from BitNetMCU import QuantizedModel, BitLinear, Activation

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
        raise ValueError(f"Model {model_name} not found in models.py or test_inference.py")


def tabular_to_cnnmnist_image(X):
    if X.shape[1] > 256:
        raise ValueError(f"Too many features for 16x16 CNN input: {X.shape[1]} > 256")

    X_pad = np.zeros((X.shape[0], 256), dtype=np.float32)
    X_pad[:, :X.shape[1]] = X
    return X_pad.reshape(-1, 1, 16, 16)


def load_gate_driver_test_excel(test_file, label_col="label", model_name="GateDriverMLP"):
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test Excel file not found: {test_file}")

    test_df = pd.read_excel(test_file)

    if label_col not in test_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in test file.")

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

    missing_cols = [c for c in feature_cols if c not in test_df.columns]

    if len(missing_cols) > 0:
        raise ValueError(f"Test file missing required feature columns: {missing_cols}")

    X_test = test_df[feature_cols].copy()

    # Use test median only for remaining NaN. Main scaling still uses train scaler.
    X_test = X_test.fillna(X_test.median(numeric_only=True))
    X_test = scaler.transform(X_test).astype(np.float32)

    input_dim = len(feature_cols)

    if model_name == "CNNMNIST":
        print("Converting gate-driver test features to CNNMNIST input shape [N, 1, 16, 16]...")
        X_test = tabular_to_cnnmnist_image(X_test)

    y_raw = test_df[label_col].astype(str)

    unknown_labels = set(y_raw.unique()) - set(label_to_id.keys())

    if len(unknown_labels) > 0:
        raise ValueError(f"Test file has labels not seen during training: {unknown_labels}")

    y_test = y_raw.map(label_to_id).values.astype(np.int64)

    test_data = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    num_classes = len(label_to_id)

    print("Gate-driver TEST Excel:", test_file)
    print("Test samples:", len(test_data))
    print("Original input dimension:", input_dim)
    print("Number of classes:", num_classes)
    print("Label mapping:", label_to_id)

    return test_data, num_classes, input_dim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate-driver CNNMNIST inference script")
    parser.add_argument(
        "--params",
        type=str,
        help="Name of the parameter file",
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
    print("Device:", device)

    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    if dataset_name != "GATE_DRIVER":
        raise ValueError("This test_inference.py is prepared for dataset: GATE_DRIVER only.")

    test_data, num_classes, input_dim = load_gate_driver_test_excel(
        test_file=hyperparameters["test_file"],
        label_col=hyperparameters.get("label_col", "label"),
        model_name=hyperparameters["model"],
    )

    hyperparameters["num_classes"] = num_classes
    hyperparameters["input_dim"] = input_dim

    test_loader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
    )

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

    print("Loading model...")

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

    print("Inference using original trained model...")

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testaccuracy = correct / total * 100
    print(f"Accuracy/Test of trained model: {testaccuracy:.2f} %")

    print("Quantizing model...")

    quantized_model = QuantizedModel(model)

    total_bits = quantized_model.totalbits()
    print(f"Total number of bits: {total_bits} ({total_bits / 8 / 1024:.4f} kbytes)")

    print("Inference using quantized Python model...")

    test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

    counter = 0
    correct_py = 0

    for input_data, labels in test_loader2:
        input_numpy = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels_numpy = labels.cpu().numpy()

        result_py = quantized_model.inference_quantized(input_numpy)
        predict_py = np.argmax(result_py, axis=1)

        if predict_py[0] == labels_numpy[0]:
            correct_py += 1

        counter += 1

    print("Size of test data:", counter)
    print(f"Mispredictions Python: {counter - correct_py}")
    print("Overall accuracy Python:", correct_py / counter * 100, "%")
