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

import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from BitNetMCU import QuantizedModel
# from models import FCMNIST
from ctypes import CDLL, c_uint32, c_int8, c_uint8, POINTER
import argparse
import yaml
import importlib
import os

# Export quantized model from saved checkpoint
# cpldcpu 2024-04-14
# Note: Hyperparameters are used to generated the filename
#---------------------------------------------

def create_run_name(hyperparameters):
    runname = hyperparameters["runtag"] + '_' + hyperparameters["model"] + ('_Aug' if hyperparameters["augmentation"] else '') + '_BitMnist_' + hyperparameters["QuantType"] + "_width" + str(hyperparameters["network_width1"]) + "_" + str(hyperparameters["network_width2"]) + "_" + str(hyperparameters["network_width3"])  + "_epochs" + str(hyperparameters["num_epochs"])
    hyperparameters["runname"] = runname
    return runname


def load_model(model_name, params):
    try:
        module = importlib.import_module('models')
        model_class = getattr(module, model_name)

        kwargs = dict(
            network_width1=params["network_width1"],
            network_width2=params["network_width2"],
            network_width3=params["network_width3"],
            QuantType=params["QuantType"],
            NormType=params["NormType"],
            WScale=params["WScale"]
        )

        if 'cnn_width' in params:
            kwargs['cnn_width'] = params['cnn_width']

        # Important for Olivetti: 40 output classes
        if 'num_classes' in params:
            kwargs['num_classes'] = params['num_classes']

        return model_class(**kwargs)

    except AttributeError:
        raise ValueError(f"Model {model_name} not found in models.py")


def export_test_data_to_c(test_loader, filename, num=8):
    with open(filename, 'w') as f:
        for i, (input_data, labels) in enumerate(test_loader):
            if i >= num:
                break

            # Reshape and convert to numpy
            input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
            labels = labels.cpu().numpy()

            scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
            scaled_data = np.round(input_data * scale).clip(-128, 127).astype(np.uint8)

            f.write(f'int8_t input_data_{i}[256] = {{\n')

            flattened_data = scaled_data.flatten()

            for k in range(0, len(flattened_data), 16):
                f.write(', '.join(f'0x{value:02X}' for value in flattened_data[k:k+16]) + ',\n')

            f.write('};\n')

            f.write(f'uint8_t label_{i} = ' + str(labels[0]) + ';\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--params', type=str, help='Name of the parameter file', default='trainingparameters.yaml')

    args = parser.parse_args()

    if args.params:
        paramname = args.params
    else:
        paramname = 'trainingparameters.yaml'

    print(f'Load parameters from file: {paramname}')

    with open(paramname) as f:
        hyperparameters = yaml.safe_load(f)

    # main
    runname = create_run_name(hyperparameters)
    print(runname)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset_name = hyperparameters.get("dataset", "MNIST").upper()

    if dataset_name == "MNIST":
        num_classes = 10
        mean, std = (0.1307,), (0.3081,)
        dataset_cls = datasets.MNIST

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_data = dataset_cls(root='data', train=True, transform=transform, download=True)
        test_data = dataset_cls(root='data', train=False, transform=transform, download=True)

    elif dataset_name == "FASHION":
        num_classes = 10
        mean, std = (0.2860,), (0.3530,)
        dataset_cls = datasets.FashionMNIST

        transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_data = dataset_cls(root='data', train=True, transform=transform, download=True)
        test_data = dataset_cls(root='data', train=False, transform=transform, download=True)

    elif dataset_name == "OLIVETTI":
        # ============================================================
        # Olivetti Face Dataset
        #
        # Same original structure:
        # train_data = training set
        # test_data  = final testing set
        #
        # This evaluates final test accuracy.
        # It does NOT use validation_set here.
        # ============================================================

        num_classes = 40
        mean, std = (0.5,), (0.5,)

        data_root = hyperparameters["data_root"]

        train_folder = hyperparameters.get(
            "train_folder",
            "olivetti_training_set_ONLY"
        )

        test_folder = hyperparameters.get(
            "test_folder",
            "olivetti_FINAL_TEST_SET_LOCKED_ONLY"
        )

        train_dir = os.path.join(data_root, train_folder)
        test_dir = os.path.join(data_root, test_folder)

        print("Olivetti train folder:", train_dir)
        print("Olivetti TEST folder:", test_dir)

        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training folder not found: {train_dir}")

        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Testing folder not found: {test_dir}")

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_data = ImageFolder(root=train_dir, transform=transform)
        test_data = ImageFolder(root=test_dir, transform=transform)

        if train_data.class_to_idx != test_data.class_to_idx:
             raise ValueError(
                 "Class mapping mismatch between training and test folders. "
                 "Please make sure both folders contain the same class_00 to class_39 structure."
             )
            
        if len(train_data.classes) != 40:
            raise ValueError(
                f"Olivetti training folder should contain 40 classes, "
                f"but found {len(train_data.classes)} classes."
            )

        if len(test_data.classes) != 40:
            raise ValueError(
                f"Olivetti test folder should contain 40 classes, "
                f"but found {len(test_data.classes)} classes."
            )
            
        print("Olivetti class mapping:", test_data.class_to_idx)
        print("Training samples:", len(train_data))
        print("Testing samples:", len(test_data))

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Important: pass number of classes to model
    hyperparameters["num_classes"] = num_classes

    # Create data loaders
    test_loader = DataLoader(
        test_data,
        batch_size=hyperparameters["batch_size"],
        shuffle=False
    )

    model = load_model(hyperparameters["model"], hyperparameters).to(device)

    # Check model output
    dummy_input = torch.randn(1, 1, 16, 16).to(device)

    with torch.no_grad():
        dummy_output = model(dummy_input)

    print("Model output shape:", dummy_output.shape)
    print("Expected classes:", num_classes)

    if dummy_output.shape[1] != num_classes:
        raise ValueError(
            f"Model output mismatch. "
            f"Model outputs {dummy_output.shape[1]} classes, "
            f"but dataset needs {num_classes} classes. "
            f"Please modify models.py final layer to use num_classes."
        )

    print('Loading model...')

    try:
        model.load_state_dict(
            torch.load(
                f'modeldata/{runname}.pth',
                map_location=torch.device('cpu')
            )
        )

    except FileNotFoundError:
        print(f"The file 'modeldata/{runname}.pth' does not exist.")
        exit()

    model = model.to(device)
    model.eval()

    print('Inference using the original model...')

    correct = 0
    total = 0
    test_loss = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    testaccuracy = correct / total * 100

    print(f'Accuracy/Test of trained model: {testaccuracy} %')

    print('Quantizing model...')

    # Quantize the model
    quantized_model = QuantizedModel(model)

    print(f'Total number of bits: {quantized_model.totalbits()} ({quantized_model.totalbits()/8/1024} kbytes)')

    # Inference using the quantized model
    print("Verifying inference of quantized model in Python and C")

    # Initialize counter
    counter = 0
    correct_c = 0
    correct_py = 0
    mismatch = 0

    test_loader2 = DataLoader(test_data, batch_size=1, shuffle=False)

    # Optional: export some final test samples to C header
    # export_test_data_to_c(test_loader2, 'BitNetMCU_Olivetti_test_data.h', num=10)

    dll_path = hyperparameters.get("dll_path", "./Bitnet_inf.dll")

    if not os.path.exists(dll_path):
        raise FileNotFoundError(
            f"Cannot find DLL file: {dll_path}\n"
            "Please make sure Bitnet_inf.dll is compiled from the exported Olivetti model."
        )

    lib = CDLL(dll_path)

    lib.Inference.argtypes = [POINTER(c_int8)]
    lib.Inference.restype = c_uint32

    for input_data, labels in test_loader2:
        input_data = input_data.view(input_data.size(0), -1).cpu().numpy()
        labels = labels.cpu().numpy()

        scale = 127.0 / np.maximum(np.abs(input_data).max(axis=-1, keepdims=True), 1e-5)
        scaled_data = np.round(input_data * scale).clip(-128, 127)

        # Create a pointer to the ctypes array
        input_data_pointer = (c_int8 * len(scaled_data.flatten()))(*scaled_data.astype(np.int8).flatten())

        # Inference C
        result_c = lib.Inference(input_data_pointer)

        # Inference Python
        result_py = quantized_model.inference_quantized(input_data)
        predict_py = np.argmax(result_py, axis=1)

        if result_c == labels[0]:
            correct_c += 1

        if predict_py[0] == labels[0]:
            correct_py += 1

        if result_c != predict_py[0]:
            print(
                f'{counter:5} Mismatch between inference engines found. '
                f'Prediction C: {result_c} '
                f'Prediction Python: {predict_py[0]} '
                f'True: {labels[0]}'
            )
            mismatch += 1

        counter += 1

    print("size of test data:", counter)
    print(f'Mispredictions C: {counter - correct_c} Py: {counter - correct_py}')
    print('Overall accuracy C:', correct_c / counter * 100, '%')
    print('Overall accuracy Python:', correct_py / counter * 100, '%')

    print(f'Mismatches between engines: {mismatch} ({mismatch/counter*100}%)')
