# import torch, torch.nn as nn, torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import ConcatDataset
# from datetime import datetime
# # from models import FCMNIST, CNNMNIST
# from BitNetMCU import BitLinear, BitConv2d, Activation
# import time
# import random
# import argparse
# import yaml
# from torchsummary import summary
# import importlib
# from models import MaskingLayer
# from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# #----------------------------------------------
# # BitNetMCU training
# #----------------------------------------------

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
#         if 'num_classes' in params:
#             kwargs['num_classes'] = params['num_classes']
#         return model_class(**kwargs)
#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py")

# def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
#     total_activations = 0
#     positive_activations = 0

#     def hook_fn(module, input, output):
#         nonlocal total_activations, positive_activations
#         if isinstance(module, nn.ReLU) or isinstance(module, Activation):
#             total_activations += output.numel()
#             positive_activations += (output > 0).sum().item()

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     # Run a forward pass to trigger hooks
#     with torch.no_grad():
#         for i in range(len(all_test_images) // batch_size):
#             images = all_test_images[i * batch_size:(i + 1) * batch_size]
#             model(images)

#     for hook in hooks:
#         hook.remove()

#     fraction_positive = positive_activations / total_activations
#     writer.add_scalar('Activations/positive_fraction', fraction_positive, epoch+1)

#     return fraction_positive


# # Function to add L1 regularization on the mask
# def add_mask_regularization(model,  lambda_l1):
#     mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

#     if mask_layer is None:
#         return 0
    
#     l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
#     return l1_reg


# def train_model(model, device, hyperparameters, train_data, test_data):
#     num_epochs = hyperparameters["num_epochs"]
#     learning_rate = hyperparameters["learning_rate"]
#     halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
#     runname =  create_run_name(hyperparameters)

#     # define dataloaders

#     batch_size = hyperparameters["batch_size"]  # Define your batch size

#     # ON-the-fly augmentation requires using the (slow) dataloader. Without augmentation, we can load the entire dataset into GPU for speedup
#     if hyperparameters["augmentation"]:
#         train_loader = DataLoader(
#         train_data, batch_size=batch_size, shuffle=True,
#         num_workers=4, pin_memory=True)
#     else:
#         # load entire dataset into GPU for 5x speedup
#         train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False) # shuffling will be done separately
#         entire_dataset = next(iter(train_loader))
#         all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     # Test dataset is always in GPU
#     test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
#     entire_dataset = next(iter(test_loader))
#     all_test_images, all_test_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     if hyperparameters["scheduler"] == "StepLR":
#         scheduler = StepLR(optimizer, step_size=hyperparameters["step_size"], gamma=hyperparameters["lr_decay"])
#     elif hyperparameters["scheduler"] == "Cosine":
#         scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)    
#     elif hyperparameters["scheduler"] == "CosineWarmRestarts":
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=hyperparameters["T_0"], T_mult=hyperparameters["T_mult"], eta_min=0)
#     else:
#         raise ValueError("Invalid scheduler")

#     criterion = nn.CrossEntropyLoss()

#     # tensorboard writer
#     now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir=f'runs/{runname}-{now_str}')

#     train_loss=[]
#     test_loss = []

#     # Train the CNN
#     for epoch in range(num_epochs):
#         all_preds = []
#         all_labels_list = []
#         all_probs = []
#         correct = 0
#         train_loss=[]
#         start_time = time.time()

#         if hyperparameters["augmentation"]:
#             for i, (images, labels) in enumerate(train_loader):
#                 images, labels = images.to(device), labels.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 loss = criterion(outputs, labels)
#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()
#         else:
#             # Shuffle images (important!)
#             indices = list(range(len(all_train_images)))
#             random.shuffle(indices)

#             for i in range(len(indices) // batch_size):
#                 batch_indices = indices[i * batch_size:(i + 1) * batch_size]
#                 images = torch.stack([all_train_images[i] for i in batch_indices])
#                 labels = torch.stack([all_train_labels[i] for i in batch_indices])
#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 loss = criterion(outputs, labels)
#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         scheduler.step()

#         if epoch + 1 == halve_lr_epoch:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= 0.5
#             print(f"Learning rate halved at epoch {epoch + 1}")


#         trainaccuracy = correct / len(train_loader.dataset) * 100

#         correct = 0
#         total = 0
#         test_loss = []
#         with torch.no_grad():
#             for i in range(len(all_test_images) // batch_size):
#                 images = all_test_images[i * batch_size:(i + 1) * batch_size]
#                 labels = all_test_labels[i * batch_size:(i + 1) * batch_size]

#                 outputs = model(images)
#                 probs = torch.softmax(outputs, dim=1)
#                 _, predicted = torch.max(probs, dim=1)
                
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels_list.extend(labels.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())
                
#                 loss = criterion(outputs, labels)
#                 test_loss.append(loss.item())
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

            
#         # Log positive activations
#         activity=log_positive_activations(model, writer, epoch, all_test_images, batch_size)

#         end_time = time.time()
#         epoch_time = end_time - start_time

#         testaccuracy = correct / total * 100


#         cm = confusion_matrix(all_labels_list, all_preds)
#         f1 = f1_score(all_labels_list, all_preds, average='macro')

#         print("Confusion Matrix:\n", cm)
#         print("F1 Score:", f1)
#         print(f'Epoch [{epoch+1}/{num_epochs}], LTrain:{np.mean(train_loss):.6f} ATrain: {trainaccuracy:.2f}% LTest:{np.mean(test_loss):.6f} ATest: {correct / total * 100:.2f}% Time[s]: {epoch_time:.2f} Act: {activity*100:.1f}% w_clip/entropy[bits]: ', end='')
#         print("\nClassification Report:\n")
#         print(classification_report(all_labels_list, all_preds))
#         try:
#             roc_auc = roc_auc_score(all_labels_list, all_probs, multi_class='ovr')
#             print("ROC AUC:", roc_auc)
#         except:
#             print("ROC not available")


#         TP = np.diag(cm)
#         FP = cm.sum(axis=0) - TP
#         FN = cm.sum(axis=1) - TP
#         TN = cm.sum() - (TP + FP + FN)

#         sensitivity = TP / (TP + FN)
#         specificity = TN / (TN + FP)

#         print("Sensitivity:", sensitivity)
#         print("Specificity:", specificity)
        
#         # update clipping scalars once per epoch
#         totalbits = 0
#         for i, layer in enumerate(model.modules()):
#             if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

#                 # update clipping scalar
#                 if epoch < hyperparameters['maxw_update_until_epoch']:
#                     layer.update_clipping_scalar(layer.weight, hyperparameters['maxw_algo'], hyperparameters['maxw_quantscale'])

#                 # calculate entropy of weights
#                 w_quant, _, _ = layer.weight_quant(layer.weight)
#                 _, counts = np.unique(w_quant.cpu().detach().numpy(), return_counts=True)
#                 probabilities = counts / np.sum(counts)
#                 entropy = -np.sum(probabilities * np.log2(probabilities))

#                 print(f'{layer.s.item():.3f}/{entropy:.2f}', end=' ')

#                 totalbits += layer.weight.numel() * layer.bpw

#         print()

#         if epoch + 1 == hyperparameters ["prune_epoch"]:
#             for m in model.modules():
#                 if isinstance(m, MaskingLayer):            
#                     pruned_channels, remaining_channels = m.prune_channels(prune_number=hyperparameters['prune_groupstoprune'], groups=hyperparameters['prune_totalgroups'])

#         writer.add_scalar('Loss/train', np.mean(train_loss), epoch+1)
#         writer.add_scalar('Accuracy/train', trainaccuracy, epoch+1)
#         writer.add_scalar('Loss/test', np.mean(test_loss), epoch+1)
#         writer.add_scalar('Accuracy/test', testaccuracy, epoch+1)
#         writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
#         writer.flush()

#     numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # totalbits = numofweights * hyperparameters['BPW']

#     print(f'TotalBits: {totalbits} TotalBytes: {totalbits/8.0} ')

#     writer.add_hparams(hyperparameters, {'Parameters': numofweights, 'Totalbits': totalbits, 'Accuracy/train': trainaccuracy, 'Accuracy/test': testaccuracy, 'Loss/train': np.mean(train_loss), 'Loss/test': np.mean(test_loss)})
#     writer.close()

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

#     runname= create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset selection (MNIST default, EMNIST optional)
#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     if dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         base_dataset_train = datasets.MNIST
#         base_dataset_test = datasets.MNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}
#     elif dataset_name.startswith("EMNIST"):
#         # Expected format: EMNIST or EMNIST_BALANCED, EMNIST_BYCLASS etc.
#         # Torchvision subsets: 'byclass'(62), 'bymerge'(47), 'balanced'(47), 'letters'(37), 'digits'(10), 'mnist'(10)
#         split = dataset_name.split('_')[1].lower() if '_' in dataset_name else 'balanced'
#         # Map common names
#         split_alias = { 'BALANCED':'balanced', 'BYCLASS':'byclass', 'BYMERGE':'bymerge', 'LETTERS':'letters', 'DIGITS':'digits', 'MNIST':'mnist'}
#         split = split_alias.get(split.upper(), split)
#         # class counts per split
#         split_classes = { 'byclass':62, 'bymerge':47, 'balanced':47, 'letters':37, 'digits':10, 'mnist':10 }
#         num_classes = split_classes.get(split, 47)
#         # EMNIST uses same normalization as MNIST typically
#         mean, std = (0.1307,), (0.3081,)
#         from torchvision.datasets import EMNIST
#         base_dataset_train = EMNIST
#         base_dataset_test = EMNIST
#         dataset_kwargs = {"split": split, "train": True}
#         dataset_kwargs_test = {"split": split, "train": False}
#     elif dataset_name == "FASHION":
#         num_classes = 10
#         mean, std = (0.5,), (0.5,)
#         base_dataset_train = datasets.FashionMNIST
#         base_dataset_test = datasets.FashionMNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     transform = transforms.Compose([
#         transforms.Resize((16, 16)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std)
#     ])

#     train_data = base_dataset_train(root='data', transform=transform, download=True, **dataset_kwargs)
#     test_data = base_dataset_test(root='data', transform=transform, download=True, **dataset_kwargs_test)

#     if hyperparameters["augmentation"]:
#         # Data augmentation for training data
#         augmented_transform = transforms.Compose([
#             transforms.RandomRotation(degrees=hyperparameters["rotation1"]),
#             transforms.RandomAffine(degrees=hyperparameters["rotation2"], translate=(0.1, 0.1), scale=(0.9, 1.1)),
#             transforms.RandomApply([
#                 transforms.ElasticTransform(alpha=40.0, sigma=4.0)
#             ], p=hyperparameters["elastictransformprobability"]),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         augmented_train_data = base_dataset_train(root='data', transform=augmented_transform, download=True, **dataset_kwargs)
#         train_data = ConcatDataset([train_data, augmented_train_data])

#     # Pass num_classes dynamically to model
#     hyperparameters['num_classes'] = num_classes
#     model = load_model(hyperparameters["model"], {**hyperparameters, 'num_classes': num_classes})
#     # If model class supports num_classes argument, it will be used. Otherwise ignore.
#     if hasattr(model, 'to'):
#         model = model.to(device)

#     summary(model, input_size=(1, 16, 16))  # Assuming the input size is (1, 16, 16)

#     print('training...')
#     train_model(model, device, hyperparameters, train_data, test_data)

#     print('saving model...')
#     torch.save(model.state_dict(), f'modeldata/{runname}.pth')

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import ConcatDataset
# from datetime import datetime
# from BitNetMCU import BitLinear, BitConv2d, Activation
# import time
# import random
# import argparse
# import yaml
# from torchsummary import summary
# import importlib
# from models import MaskingLayer
# from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# #----------------------------------------------
# # BitNetMCU training
# #----------------------------------------------

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

#         if 'num_classes' in params:
#             kwargs['num_classes'] = params['num_classes']

#         return model_class(**kwargs)

#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py")


# def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
#     total_activations = 0
#     positive_activations = 0

#     def hook_fn(module, input, output):
#         nonlocal total_activations, positive_activations
#         if isinstance(module, nn.ReLU) or isinstance(module, Activation):
#             total_activations += output.numel()
#             positive_activations += (output > 0).sum().item()

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     # Run a forward pass to trigger hooks
#     with torch.no_grad():
#         for i in range(0, len(all_test_images), batch_size):
#             images = all_test_images[i:i + batch_size]
#             model(images)

#     for hook in hooks:
#         hook.remove()

#     if total_activations == 0:
#         fraction_positive = 0.0
#     else:
#         fraction_positive = positive_activations / total_activations

#     writer.add_scalar('Activations/positive_fraction', fraction_positive, epoch+1)

#     return fraction_positive


# # Function to add L1 regularization on the mask
# def add_mask_regularization(model, lambda_l1):
#     mask_layer = next((layer for layer in model.modules() if isinstance(layer, MaskingLayer)), None)

#     if mask_layer is None:
#         return 0

#     l1_reg = lambda_l1 * torch.norm(mask_layer.mask, 1)
#     return l1_reg


# def train_model(model, device, hyperparameters, train_data, test_data):
#     num_epochs = hyperparameters["num_epochs"]
#     learning_rate = hyperparameters["learning_rate"]
#     halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
#     runname = create_run_name(hyperparameters)

#     batch_size = hyperparameters["batch_size"]

#     # ON-the-fly augmentation requires using the dataloader.
#     # Without augmentation, we can load the entire dataset into GPU for speedup.
#     if hyperparameters["augmentation"]:
#         train_loader = DataLoader(
#             train_data,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#     else:
#         train_loader = DataLoader(
#             train_data,
#             batch_size=len(train_data),
#             shuffle=False
#         )

#         entire_dataset = next(iter(train_loader))
#         all_train_images, all_train_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     # Validation dataset is always loaded here.
#     # Note: test_data means validation_set for OLIVETTI.
#     test_loader = DataLoader(
#         test_data,
#         batch_size=len(test_data),
#         shuffle=False
#     )

#     entire_dataset = next(iter(test_loader))
#     all_test_images, all_test_labels = entire_dataset[0].to(device), entire_dataset[1].to(device)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     if hyperparameters["scheduler"] == "StepLR":
#         scheduler = StepLR(
#             optimizer,
#             step_size=hyperparameters["step_size"],
#             gamma=hyperparameters["lr_decay"]
#         )

#     elif hyperparameters["scheduler"] == "Cosine":
#         scheduler = CosineAnnealingLR(
#             optimizer,
#             T_max=num_epochs,
#             eta_min=0
#         )

#     elif hyperparameters["scheduler"] == "CosineWarmRestarts":
#         scheduler = CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=hyperparameters["T_0"],
#             T_mult=hyperparameters["T_mult"],
#             eta_min=0
#         )

#     else:
#         raise ValueError("Invalid scheduler")

#     criterion = nn.CrossEntropyLoss()

#     now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir=f'runs/{runname}-{now_str}')

#     train_loss = []
#     test_loss = []

#     best_testaccuracy = 0.0

#     # Train the CNN
#     for epoch in range(num_epochs):
#         model.train()

#         all_preds = []
#         all_labels_list = []
#         all_probs = []

#         correct = 0
#         train_loss = []
#         start_time = time.time()

#         if hyperparameters["augmentation"]:
#             for i, (images, labels) in enumerate(train_loader):
#                 images, labels = images.to(device), labels.to(device)

#                 optimizer.zero_grad()

#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 loss = criterion(outputs, labels)

#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

#                 loss.backward()
#                 optimizer.step()

#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         else:
#             # Shuffle images
#             indices = list(range(len(all_train_images)))
#             random.shuffle(indices)

#             for i in range(0, len(indices), batch_size):
#                 batch_indices = indices[i:i + batch_size]

#                 images = torch.stack([all_train_images[j] for j in batch_indices])
#                 labels = torch.stack([all_train_labels[j] for j in batch_indices])

#                 optimizer.zero_grad()

#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 loss = criterion(outputs, labels)

#                 if epoch < hyperparameters['prune_epoch']:
#                     loss += add_mask_regularization(model, hyperparameters["lambda_l1"])

#                 loss.backward()
#                 optimizer.step()

#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         scheduler.step()

#         if epoch + 1 == halve_lr_epoch:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= 0.5
#             print(f"Learning rate halved at epoch {epoch + 1}")

#         trainaccuracy = correct / len(train_loader.dataset) * 100

#         model.eval()

#         correct = 0
#         total = 0
#         test_loss = []

#         with torch.no_grad():
#             for i in range(0, len(all_test_images), batch_size):
#                 images = all_test_images[i:i + batch_size]
#                 labels = all_test_labels[i:i + batch_size]

#                 outputs = model(images)
#                 probs = torch.softmax(outputs, dim=1)
#                 _, predicted = torch.max(probs, dim=1)

#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels_list.extend(labels.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())

#                 loss = criterion(outputs, labels)
#                 test_loss.append(loss.item())

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         # Log positive activations
#         activity = log_positive_activations(
#             model,
#             writer,
#             epoch,
#             all_test_images,
#             batch_size
#         )

#         end_time = time.time()
#         epoch_time = end_time - start_time

#         testaccuracy = correct / total * 100

#         cm = confusion_matrix(all_labels_list, all_preds)

#         f1 = f1_score(
#             all_labels_list,
#             all_preds,
#             average='macro',
#             zero_division=0
#         )

#         print("Confusion Matrix:\n", cm)
#         print("F1 Score:", f1)

#         print(
#             f'Epoch [{epoch+1}/{num_epochs}], '
#             f'LTrain:{np.mean(train_loss):.6f} '
#             f'ATrain: {trainaccuracy:.2f}% '
#             f'LTest:{np.mean(test_loss):.6f} '
#             f'ATest: {testaccuracy:.2f}% '
#             f'Time[s]: {epoch_time:.2f} '
#             f'Act: {activity*100:.1f}% '
#             f'w_clip/entropy[bits]: ',
#             end=''
#         )

#         print("\nClassification Report:\n")
#         print(
#             classification_report(
#                 all_labels_list,
#                 all_preds,
#                 zero_division=0
#             )
#         )

#         try:
#             roc_auc = roc_auc_score(
#                 all_labels_list,
#                 all_probs,
#                 multi_class='ovr'
#             )
#             print("ROC AUC:", roc_auc)
#         except Exception:
#             print("ROC not available")

#         TP = np.diag(cm)
#         FP = cm.sum(axis=0) - TP
#         FN = cm.sum(axis=1) - TP
#         TN = cm.sum() - (TP + FP + FN)

#         sensitivity = np.divide(
#             TP,
#             TP + FN,
#             out=np.zeros_like(TP, dtype=float),
#             where=(TP + FN) != 0
#         )

#         specificity = np.divide(
#             TN,
#             TN + FP,
#             out=np.zeros_like(TN, dtype=float),
#             where=(TN + FP) != 0
#         )

#         print("Sensitivity:", sensitivity)
#         print("Specificity:", specificity)

#         # update clipping scalars once per epoch
#         totalbits = 0

#         for i, layer in enumerate(model.modules()):
#             if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

#                 # update clipping scalar
#                 if epoch < hyperparameters['maxw_update_until_epoch']:
#                     layer.update_clipping_scalar(
#                         layer.weight,
#                         hyperparameters['maxw_algo'],
#                         hyperparameters['maxw_quantscale']
#                     )

#                 # calculate entropy of weights
#                 try:
#                     w_quant, _, _ = layer.weight_quant(layer.weight)

#                     _, counts = np.unique(
#                         w_quant.cpu().detach().numpy(),
#                         return_counts=True
#                     )

#                     probabilities = counts / np.sum(counts)
#                     entropy = -np.sum(probabilities * np.log2(probabilities))

#                     print(f'{layer.s.item():.3f}/{entropy:.2f}', end=' ')

#                     totalbits += layer.weight.numel() * layer.bpw

#                 except Exception:
#                     pass

#         print()

#         if epoch + 1 == hyperparameters["prune_epoch"]:
#             for m in model.modules():
#                 if isinstance(m, MaskingLayer):
#                     pruned_channels, remaining_channels = m.prune_channels(
#                         prune_number=hyperparameters['prune_groupstoprune'],
#                         groups=hyperparameters['prune_totalgroups']
#                     )

#         writer.add_scalar('Loss/train', np.mean(train_loss), epoch+1)
#         writer.add_scalar('Accuracy/train', trainaccuracy, epoch+1)
#         writer.add_scalar('Loss/test', np.mean(test_loss), epoch+1)
#         writer.add_scalar('Accuracy/test', testaccuracy, epoch+1)
#         writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
#         writer.flush()

#         # Save best validation model
#         if testaccuracy > best_testaccuracy:
#             best_testaccuracy = testaccuracy
#             os.makedirs("modeldata", exist_ok=True)
#             torch.save(model.state_dict(), f'modeldata/{runname}_best.pth')
#             print(f"Best model saved. Best validation accuracy: {best_testaccuracy:.2f}%")

#     numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f'TotalBits: {totalbits} TotalBytes: {totalbits/8.0} ')

#     writer.add_hparams(
#         hyperparameters,
#         {
#             'Parameters': numofweights,
#             'Totalbits': totalbits,
#             'Accuracy/train': trainaccuracy,
#             'Accuracy/test': testaccuracy,
#             'Loss/train': np.mean(train_loss),
#             'Loss/test': np.mean(test_loss)
#         }
#     )

#     writer.close()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Training script')
#     parser.add_argument(
#         '--params',
#         type=str,
#         help='Name of the parameter file',
#         default='trainingparameters.yaml'
#     )

#     args = parser.parse_args()

#     if args.params:
#         paramname = args.params
#     else:
#         paramname = 'trainingparameters.yaml'

#     print(f'Load parameters from file: {paramname}')

#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     runname = create_run_name(hyperparameters)
#     print(runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset selection
#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     if dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         base_dataset_train = datasets.MNIST
#         base_dataset_test = datasets.MNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     elif dataset_name.startswith("EMNIST"):
#         # Expected format:
#         # EMNIST_BALANCED, EMNIST_BYCLASS, EMNIST_BYMERGE,
#         # EMNIST_LETTERS, EMNIST_DIGITS, EMNIST_MNIST

#         split = dataset_name.split('_')[1].lower() if '_' in dataset_name else 'balanced'

#         split_alias = {
#             'BALANCED': 'balanced',
#             'BYCLASS': 'byclass',
#             'BYMERGE': 'bymerge',
#             'LETTERS': 'letters',
#             'DIGITS': 'digits',
#             'MNIST': 'mnist'
#         }

#         split = split_alias.get(split.upper(), split)

#         split_classes = {
#             'byclass': 62,
#             'bymerge': 47,
#             'balanced': 47,
#             'letters': 37,
#             'digits': 10,
#             'mnist': 10
#         }

#         num_classes = split_classes.get(split, 47)

#         mean, std = (0.1307,), (0.3081,)

#         from torchvision.datasets import EMNIST

#         base_dataset_train = EMNIST
#         base_dataset_test = EMNIST

#         dataset_kwargs = {
#             "split": split,
#             "train": True
#         }

#         dataset_kwargs_test = {
#             "split": split,
#             "train": False
#         }

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     elif dataset_name == "FASHION":
#         num_classes = 10
#         mean, std = (0.5,), (0.5,)
#         base_dataset_train = datasets.FashionMNIST
#         base_dataset_test = datasets.FashionMNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root='data',
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     elif dataset_name == "OLIVETTI":
#         # ============================================================
#         # OLIVETTI DATASET
#         #
#         # training_set   -> train_data
#         # validation_set -> test_data
#         #
#         # final_testing_set_LOCKED is NOT used here.
#         # ============================================================

#         num_classes = 40
#         mean, std = (0.5,), (0.5,)

#         data_root = hyperparameters["data_root"]

#         train_folder = hyperparameters.get(
#             "train_folder",
#             "olivetti_training_set_ONLY"
#         )

#         val_folder = hyperparameters.get(
#             "val_folder",
#             "olivetti_validation_set_ONLY"
#         )

#         train_dir = os.path.join(data_root, train_folder)
#         val_dir = os.path.join(data_root, val_folder)

#         print("Olivetti train folder:", train_dir)
#         print("Olivetti validation folder:", val_dir)

#         if not os.path.exists(train_dir):
#             raise FileNotFoundError(f"Training folder not found: {train_dir}")

#         if not os.path.exists(val_dir):
#             raise FileNotFoundError(f"Validation folder not found: {val_dir}")

#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = ImageFolder(
#             root=train_dir,
#             transform=transform
#         )

#         test_data = ImageFolder(
#             root=val_dir,
#             transform=transform
#         )

#         if train_data.class_to_idx != test_data.class_to_idx:
#             raise ValueError(
#                 "Class mapping mismatch between training and validation folders. "
#                 "Please make sure both folders contain the same class_00 to class_39 structure."
#             )

#         if len(train_data.classes) != 40:
#             raise ValueError(
#                  f"Olivetti training folder should contain 40 classes, "
#                  f"but found {len(train_data.classes)} classes."
#              )

#         if len(test_data.classes) != 40:
#             raise ValueError(
#                 f"Olivetti validation folder should contain 40 classes, "
#                 f"but found {len(test_data.classes)} classes."
#             )   
            
#         print("Olivetti class mapping:", train_data.class_to_idx)
#         print("Training samples:", len(train_data))
#         print("Validation samples:", len(test_data))

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     if hyperparameters["augmentation"]:
#         # Data augmentation for training data
#         augmented_transform = transforms.Compose([
#             transforms.RandomRotation(degrees=hyperparameters["rotation1"]),

#             transforms.RandomAffine(
#                 degrees=hyperparameters["rotation2"],
#                 translate=(0.1, 0.1),
#                 scale=(0.9, 1.1)
#             ),

#             transforms.RandomApply([
#                 transforms.ElasticTransform(alpha=40.0, sigma=4.0)
#             ], p=hyperparameters["elastictransformprobability"]),

#             transforms.Resize((16, 16)),

#             # Needed for Olivetti ImageFolder PNG images
#             transforms.Grayscale(num_output_channels=1)
#             if dataset_name == "OLIVETTI"
#             else transforms.Lambda(lambda x: x),

#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         if dataset_name == "OLIVETTI":
#             augmented_train_data = ImageFolder(
#                 root=train_dir,
#                 transform=augmented_transform
#             )
#         else:
#             augmented_train_data = base_dataset_train(
#                 root='data',
#                 transform=augmented_transform,
#                 download=True,
#                 **dataset_kwargs
#             )

#         train_data = ConcatDataset([train_data, augmented_train_data])

#     # Pass num_classes dynamically to model
#     hyperparameters['num_classes'] = num_classes

#     model = load_model(
#         hyperparameters["model"],
#         {**hyperparameters, 'num_classes': num_classes}
#     )

#     # If model class supports num_classes argument, it will be used.
#     if hasattr(model, 'to'):
#         model = model.to(device)

#     # Check output size before training
#     dummy_input = torch.randn(1, 1, 16, 16).to(device)

#     with torch.no_grad():
#         dummy_output = model(dummy_input)

#     print("Model output shape:", dummy_output.shape)
#     print("Expected output classes:", num_classes)

#     if dummy_output.shape[1] != num_classes:
#         raise ValueError(
#             f"Model output class mismatch. "
#             f"Model outputs {dummy_output.shape[1]} classes, "
#             f"but dataset needs {num_classes} classes. "
#             f"Please modify models.py final layer to use num_classes."
#         )

#     summary(model, input_size=(1, 16, 16))

#     print('training...')
#     train_model(model, device, hyperparameters, train_data, test_data)

#     print('saving model...')
#     os.makedirs("modeldata", exist_ok=True)
#     torch.save(model.state_dict(), f'modeldata/{runname}.pth')

#     print("Model saved to:", f'modeldata/{runname}.pth')

# import os
# import time
# import json
# import pickle
# import random
# import argparse
# from datetime import datetime

# import yaml
# import numpy as np
# import pandas as pd

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from torchvision import datasets, transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

# from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
# from torch.utils.tensorboard import SummaryWriter

# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, f1_score, classification_report, roc_auc_score

# try:
#     from torchsummary import summary
# except Exception:
#     summary = None

# from BitNetMCU import BitLinear, BitConv2d, Activation

# try:
#     import importlib
#     from models import MaskingLayer
# except Exception:
#     MaskingLayer = None


# # ============================================================
# # BitNetMCU Gate Driver MLP Model
# # For Excel/tabular synthetic gate-driver dataset
# # ============================================================

# def make_bitlinear(in_features, out_features, QuantType, NormType, WScale):
#     """
#     Robust BitLinear creator.
#     Works with most BitNetMCU BitLinear definitions.
#     """
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
#             print("Warning: BitLinear argument mismatch. Falling back to nn.Linear.")
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
#         if dropout > 0:
#             layers.append(nn.Dropout(dropout))

#         layers.append(make_bitlinear(network_width1, network_width2, QuantType, NormType, WScale))
#         layers.append(Activation())
#         if dropout > 0:
#             layers.append(nn.Dropout(dropout))

#         if network_width3 and network_width3 > 0:
#             layers.append(make_bitlinear(network_width2, network_width3, QuantType, NormType, WScale))
#             layers.append(Activation())
#             if dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#             layers.append(make_bitlinear(network_width3, num_classes, QuantType, NormType, WScale))
#         else:
#             layers.append(make_bitlinear(network_width2, num_classes, QuantType, NormType, WScale))

#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)


# # ============================================================
# # Utility functions
# # ============================================================

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


# def load_model(model_name, params):
#     """
#     Loads GateDriverMLP from this file.
#     Otherwise loads image models from models.py.
#     """

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
#             WScale=params["WScale"],
#         )

#         if "cnn_width" in params:
#             kwargs["cnn_width"] = params["cnn_width"]

#         if "num_classes" in params:
#             kwargs["num_classes"] = params["num_classes"]

#         return model_class(**kwargs)

#     except AttributeError:
#         raise ValueError(f"Model {model_name} not found in models.py or training.py")


# def add_mask_regularization(model, lambda_l1):
#     if MaskingLayer is None:
#         return 0

#     mask_layer = next(
#         (layer for layer in model.modules() if isinstance(layer, MaskingLayer)),
#         None
#     )

#     if mask_layer is None:
#         return 0

#     return lambda_l1 * torch.norm(mask_layer.mask, 1)


# def log_positive_activations(model, writer, epoch, all_test_images, batch_size):
#     total_activations = 0
#     positive_activations = 0

#     def hook_fn(module, input, output):
#         nonlocal total_activations, positive_activations
#         if isinstance(module, nn.ReLU) or isinstance(module, Activation):
#             total_activations += output.numel()
#             positive_activations += (output > 0).sum().item()

#     hooks = []
#     for layer in model.modules():
#         if isinstance(layer, nn.ReLU) or isinstance(layer, Activation):
#             hooks.append(layer.register_forward_hook(hook_fn))

#     with torch.no_grad():
#         for i in range(0, len(all_test_images), batch_size):
#             images = all_test_images[i:i + batch_size]
#             model(images)

#     for hook in hooks:
#         hook.remove()

#     if total_activations == 0:
#         fraction_positive = 0.0
#     else:
#         fraction_positive = positive_activations / total_activations

#     writer.add_scalar("Activations/positive_fraction", fraction_positive, epoch + 1)
#     return fraction_positive


# # ============================================================
# # Excel Gate Driver Dataset Loader
# # ============================================================

# def load_gate_driver_excel(train_file, val_file, label_col="label"):
#     """
#     Loads Excel gate-driver dataset safely.

#     Removes leakage columns:
#     - label / label_id
#     - sample_id
#     - class/fault names
#     - flag_* protection outputs

#     Uses only numeric measurement columns as input.
#     """

#     train_df = pd.read_excel(train_file)
#     val_df = pd.read_excel(val_file)

#     print("Train Excel shape:", train_df.shape)
#     print("Validation Excel shape:", val_df.shape)

#     if label_col not in train_df.columns:
#         raise ValueError(f"Label column '{label_col}' not found in train file.")

#     if label_col not in val_df.columns:
#         raise ValueError(f"Label column '{label_col}' not found in validation file.")

#     leakage_cols = {
#         "label",
#         "label_id",
#         "sample_id",
#         "fault_name",
#         "class_name",
#         "class",
#         "fault",
#         "target",
#         "y",
#         "split",
#     }

#     # Remove exact leakage columns and all flag_* columns
#     feature_cols = []
#     for c in train_df.columns:
#         c_lower = c.lower()

#         if c_lower in leakage_cols:
#             continue

#         if c_lower.startswith("flag_"):
#             continue

#         if "label" in c_lower:
#             continue

#         if "fault" in c_lower and c_lower != "fault_current":
#             continue

#         if pd.api.types.is_numeric_dtype(train_df[c]):
#             feature_cols.append(c)

#     if len(feature_cols) == 0:
#         raise ValueError("No safe numeric input features found after leakage removal.")

#     print("\nSafe input features used by the model:")
#     for c in feature_cols:
#         print(" -", c)

#     # Label mapping from train file
#     train_labels_raw = train_df[label_col].astype(str)
#     val_labels_raw = val_df[label_col].astype(str)

#     label_names = sorted(train_labels_raw.unique().tolist())
#     label_to_id = {name: idx for idx, name in enumerate(label_names)}

#     unknown_val_labels = set(val_labels_raw.unique()) - set(label_to_id.keys())
#     if len(unknown_val_labels) > 0:
#         raise ValueError(f"Validation has labels not found in train: {unknown_val_labels}")

#     y_train = train_labels_raw.map(label_to_id).values.astype(np.int64)
#     y_val = val_labels_raw.map(label_to_id).values.astype(np.int64)

#     X_train = train_df[feature_cols].copy()
#     X_val = val_df[feature_cols].copy()

#     # Fill missing values using train median only
#     medians = X_train.median(numeric_only=True)
#     X_train = X_train.fillna(medians)
#     X_val = X_val.fillna(medians)

#     X_train = X_train.values.astype(np.float32)
#     X_val = X_val.values.astype(np.float32)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train).astype(np.float32)
#     X_val = scaler.transform(X_val).astype(np.float32)

#     train_data = TensorDataset(
#         torch.tensor(X_train, dtype=torch.float32),
#         torch.tensor(y_train, dtype=torch.long)
#     )

#     val_data = TensorDataset(
#         torch.tensor(X_val, dtype=torch.float32),
#         torch.tensor(y_val, dtype=torch.long)
#     )

#     num_classes = len(label_names)
#     input_dim = len(feature_cols)

#     os.makedirs("modeldata", exist_ok=True)

#     with open("modeldata/gate_driver_feature_cols.json", "w") as f:
#         json.dump(feature_cols, f, indent=2)

#     with open("modeldata/gate_driver_label_mapping.json", "w") as f:
#         json.dump(label_to_id, f, indent=2)

#     with open("modeldata/gate_driver_scaler.pkl", "wb") as f:
#         pickle.dump(scaler, f)

#     print("\nLabel mapping:")
#     print(label_to_id)
#     print("Input dimension:", input_dim)
#     print("Number of classes:", num_classes)

#     return train_data, val_data, num_classes, input_dim, feature_cols


# # ============================================================
# # Training function
# # ============================================================

# def train_model(model, device, hyperparameters, train_data, test_data):
#     num_epochs = hyperparameters["num_epochs"]
#     learning_rate = hyperparameters["learning_rate"]
#     halve_lr_epoch = hyperparameters.get("halve_lr_epoch", -1)
#     runname = create_run_name(hyperparameters)
#     batch_size = hyperparameters["batch_size"]

#     if hyperparameters.get("augmentation", False):
#         train_loader = DataLoader(
#             train_data,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=0,
#             pin_memory=True
#         )
#     else:
#         train_loader = DataLoader(
#             train_data,
#             batch_size=len(train_data),
#             shuffle=False
#         )

#         entire_dataset = next(iter(train_loader))
#         all_train_images = entire_dataset[0].to(device)
#         all_train_labels = entire_dataset[1].to(device)

#     test_loader = DataLoader(
#         test_data,
#         batch_size=len(test_data),
#         shuffle=False
#     )

#     entire_dataset = next(iter(test_loader))
#     all_test_images = entire_dataset[0].to(device)
#     all_test_labels = entire_dataset[1].to(device)

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=hyperparameters.get("weight_decay", 0.0)
#     )

#     scheduler_name = hyperparameters.get("scheduler", "Cosine")

#     if scheduler_name == "StepLR":
#         scheduler = StepLR(
#             optimizer,
#             step_size=hyperparameters.get("step_size", 20),
#             gamma=hyperparameters.get("lr_decay", 0.5)
#         )

#     elif scheduler_name == "Cosine":
#         scheduler = CosineAnnealingLR(
#             optimizer,
#             T_max=num_epochs,
#             eta_min=0
#         )

#     elif scheduler_name == "CosineWarmRestarts":
#         scheduler = CosineAnnealingWarmRestarts(
#             optimizer,
#             T_0=hyperparameters.get("T_0", 10),
#             T_mult=hyperparameters.get("T_mult", 2),
#             eta_min=0
#         )

#     else:
#         raise ValueError("Invalid scheduler")

#     criterion = nn.CrossEntropyLoss()

#     now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
#     writer = SummaryWriter(log_dir=f"runs/{runname}-{now_str}")

#     best_testaccuracy = 0.0
#     best_f1 = 0.0
#     totalbits = 0

#     for epoch in range(num_epochs):
#         model.train()

#         correct = 0
#         train_loss = []
#         start_time = time.time()

#         if hyperparameters.get("augmentation", False):
#             for images, labels in train_loader:
#                 images, labels = images.to(device), labels.to(device)

#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 loss = criterion(outputs, labels)

#                 if epoch < hyperparameters.get("prune_epoch", -1):
#                     loss += add_mask_regularization(model, hyperparameters.get("lambda_l1", 0.0))

#                 loss.backward()
#                 optimizer.step()

#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         else:
#             indices = list(range(len(all_train_images)))
#             random.shuffle(indices)

#             for i in range(0, len(indices), batch_size):
#                 batch_indices = indices[i:i + batch_size]

#                 images = all_train_images[batch_indices]
#                 labels = all_train_labels[batch_indices]

#                 optimizer.zero_grad()
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)

#                 loss = criterion(outputs, labels)

#                 if epoch < hyperparameters.get("prune_epoch", -1):
#                     loss += add_mask_regularization(model, hyperparameters.get("lambda_l1", 0.0))

#                 loss.backward()
#                 optimizer.step()

#                 train_loss.append(loss.item())
#                 correct += (predicted == labels).sum().item()

#         scheduler.step()

#         if epoch + 1 == halve_lr_epoch:
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] *= 0.5
#             print(f"Learning rate halved at epoch {epoch + 1}")

#         trainaccuracy = correct / len(train_loader.dataset) * 100

#         model.eval()

#         correct = 0
#         total = 0
#         test_loss = []

#         all_preds = []
#         all_labels_list = []
#         all_probs = []

#         with torch.no_grad():
#             for i in range(0, len(all_test_images), batch_size):
#                 images = all_test_images[i:i + batch_size]
#                 labels = all_test_labels[i:i + batch_size]

#                 outputs = model(images)
#                 probs = torch.softmax(outputs, dim=1)
#                 _, predicted = torch.max(probs, dim=1)

#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels_list.extend(labels.cpu().numpy())
#                 all_probs.extend(probs.cpu().numpy())

#                 loss = criterion(outputs, labels)
#                 test_loss.append(loss.item())

#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         activity = log_positive_activations(
#             model,
#             writer,
#             epoch,
#             all_test_images,
#             batch_size
#         )

#         epoch_time = time.time() - start_time
#         testaccuracy = correct / total * 100

#         cm = confusion_matrix(all_labels_list, all_preds)

#         f1 = f1_score(
#             all_labels_list,
#             all_preds,
#             average="macro",
#             zero_division=0
#         )

#         print("\nConfusion Matrix:\n", cm)
#         print("F1 Score:", f1)

#         print(
#             f"Epoch [{epoch + 1}/{num_epochs}], "
#             f"LTrain:{np.mean(train_loss):.6f} "
#             f"ATrain:{trainaccuracy:.2f}% "
#             f"LVal:{np.mean(test_loss):.6f} "
#             f"AVal:{testaccuracy:.2f}% "
#             f"Time[s]:{epoch_time:.2f} "
#             f"Act:{activity * 100:.1f}% "
#             f"w_clip/entropy[bits]: ",
#             end=""
#         )

#         print("\nClassification Report:\n")
#         print(
#             classification_report(
#                 all_labels_list,
#                 all_preds,
#                 zero_division=0
#             )
#         )

#         try:
#             roc_auc = roc_auc_score(
#                 all_labels_list,
#                 np.array(all_probs),
#                 multi_class="ovr"
#             )
#             print("ROC AUC:", roc_auc)
#         except Exception:
#             print("ROC not available")

#         TP = np.diag(cm)
#         FP = cm.sum(axis=0) - TP
#         FN = cm.sum(axis=1) - TP
#         TN = cm.sum() - (TP + FP + FN)

#         sensitivity = np.divide(
#             TP,
#             TP + FN,
#             out=np.zeros_like(TP, dtype=float),
#             where=(TP + FN) != 0
#         )

#         specificity = np.divide(
#             TN,
#             TN + FP,
#             out=np.zeros_like(TN, dtype=float),
#             where=(TN + FP) != 0
#         )

#         print("Sensitivity:", sensitivity)
#         print("Specificity:", specificity)

#         totalbits = 0

#         for layer in model.modules():
#             if isinstance(layer, BitLinear) or isinstance(layer, BitConv2d):

#                 if epoch < hyperparameters.get("maxw_update_until_epoch", 60):
#                     try:
#                         layer.update_clipping_scalar(
#                             layer.weight,
#                             hyperparameters.get("maxw_algo", "octav"),
#                             hyperparameters.get("maxw_quantscale", 0.25)
#                         )
#                     except Exception:
#                         pass

#                 try:
#                     w_quant, _, _ = layer.weight_quant(layer.weight)

#                     _, counts = np.unique(
#                         w_quant.cpu().detach().numpy(),
#                         return_counts=True
#                     )

#                     probabilities = counts / np.sum(counts)
#                     entropy = -np.sum(probabilities * np.log2(probabilities))

#                     print(f"{layer.s.item():.3f}/{entropy:.2f}", end=" ")

#                     totalbits += layer.weight.numel() * layer.bpw

#                 except Exception:
#                     pass

#         print()

#         if epoch + 1 == hyperparameters.get("prune_epoch", -1):
#             if MaskingLayer is not None:
#                 for m in model.modules():
#                     if isinstance(m, MaskingLayer):
#                         m.prune_channels(
#                             prune_number=hyperparameters.get("prune_groupstoprune", 0),
#                             groups=hyperparameters.get("prune_totalgroups", 0)
#                         )

#         writer.add_scalar("Loss/train", np.mean(train_loss), epoch + 1)
#         writer.add_scalar("Accuracy/train", trainaccuracy, epoch + 1)
#         writer.add_scalar("Loss/val", np.mean(test_loss), epoch + 1)
#         writer.add_scalar("Accuracy/val", testaccuracy, epoch + 1)
#         writer.add_scalar("F1/val_macro", f1, epoch + 1)
#         writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
#         writer.flush()

#         if testaccuracy > best_testaccuracy:
#             best_testaccuracy = testaccuracy
#             best_f1 = f1
#             os.makedirs("modeldata", exist_ok=True)
#             torch.save(model.state_dict(), f"modeldata/{runname}_best.pth")
#             print(f"Best model saved. Best validation accuracy: {best_testaccuracy:.2f}%")

#     numofweights = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"\nTotalBits: {totalbits} TotalBytes: {totalbits / 8.0}")
#     print(f"Trainable Parameters: {numofweights}")
#     print(f"Best Validation Accuracy: {best_testaccuracy:.2f}%")
#     print(f"Best Validation F1 Macro: {best_f1:.4f}")

#     writer.add_hparams(
#         hyperparameters,
#         {
#             "Parameters": numofweights,
#             "Totalbits": totalbits,
#             "Accuracy/train": trainaccuracy,
#             "Accuracy/val": testaccuracy,
#             "F1/val_macro": f1,
#             "Loss/train": np.mean(train_loss),
#             "Loss/val": np.mean(test_loss),
#         }
#     )

#     writer.close()


# # ============================================================
# # Main
# # ============================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="BitNetMCU Training Script")
#     parser.add_argument(
#         "--params",
#         type=str,
#         help="Name of the parameter YAML file",
#         default="trainingparameters.yaml"
#     )

#     args = parser.parse_args()
#     paramname = args.params

#     print(f"Load parameters from file: {paramname}")

#     with open(paramname) as f:
#         hyperparameters = yaml.safe_load(f)

#     # Safe defaults
#     hyperparameters.setdefault("augmentation", False)
#     hyperparameters.setdefault("dropout", 0.1)
#     hyperparameters.setdefault("weight_decay", 0.0)
#     hyperparameters.setdefault("lambda_l1", 0.0)
#     hyperparameters.setdefault("prune_epoch", -1)
#     hyperparameters.setdefault("maxw_update_until_epoch", 60)
#     hyperparameters.setdefault("maxw_algo", "octav")
#     hyperparameters.setdefault("maxw_quantscale", 0.25)
#     hyperparameters.setdefault("network_width3", 0)

#     runname = create_run_name(hyperparameters)
#     print("Run name:", runname)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     dataset_name = hyperparameters.get("dataset", "MNIST").upper()

#     input_dim = None
#     feature_cols = None

#     # ============================================================
#     # GATE DRIVER EXCEL DATASET
#     # ============================================================

#     if dataset_name == "GATE_DRIVER":
#         train_data, test_data, num_classes, input_dim, feature_cols = load_gate_driver_excel(
#             train_file=hyperparameters["train_file"],
#             val_file=hyperparameters["val_file"],
#             label_col=hyperparameters.get("label_col", "label")
#         )

#         hyperparameters["input_dim"] = input_dim

#     # ============================================================
#     # MNIST
#     # ============================================================

#     elif dataset_name == "MNIST":
#         num_classes = 10
#         mean, std = (0.1307,), (0.3081,)
#         base_dataset_train = datasets.MNIST
#         base_dataset_test = datasets.MNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     # ============================================================
#     # EMNIST
#     # ============================================================

#     elif dataset_name.startswith("EMNIST"):
#         split = dataset_name.split("_")[1].lower() if "_" in dataset_name else "balanced"

#         split_alias = {
#             "BALANCED": "balanced",
#             "BYCLASS": "byclass",
#             "BYMERGE": "bymerge",
#             "LETTERS": "letters",
#             "DIGITS": "digits",
#             "MNIST": "mnist"
#         }

#         split = split_alias.get(split.upper(), split)

#         split_classes = {
#             "byclass": 62,
#             "bymerge": 47,
#             "balanced": 47,
#             "letters": 37,
#             "digits": 10,
#             "mnist": 10
#         }

#         num_classes = split_classes.get(split, 47)
#         mean, std = (0.1307,), (0.3081,)

#         from torchvision.datasets import EMNIST

#         base_dataset_train = EMNIST
#         base_dataset_test = EMNIST

#         dataset_kwargs = {
#             "split": split,
#             "train": True
#         }

#         dataset_kwargs_test = {
#             "split": split,
#             "train": False
#         }

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     # ============================================================
#     # FASHION MNIST
#     # ============================================================

#     elif dataset_name == "FASHION":
#         num_classes = 10
#         mean, std = (0.5,), (0.5,)
#         base_dataset_train = datasets.FashionMNIST
#         base_dataset_test = datasets.FashionMNIST
#         dataset_kwargs = {"train": True}
#         dataset_kwargs_test = {"train": False}

#         transform = transforms.Compose([
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = base_dataset_train(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs
#         )

#         test_data = base_dataset_test(
#             root="data",
#             transform=transform,
#             download=True,
#             **dataset_kwargs_test
#         )

#     # ============================================================
#     # OLIVETTI
#     # ============================================================

#     elif dataset_name == "OLIVETTI":
#         num_classes = 40
#         mean, std = (0.5,), (0.5,)

#         data_root = hyperparameters["data_root"]

#         train_folder = hyperparameters.get(
#             "train_folder",
#             "olivetti_training_set_ONLY"
#         )

#         val_folder = hyperparameters.get(
#             "val_folder",
#             "olivetti_validation_set_ONLY"
#         )

#         train_dir = os.path.join(data_root, train_folder)
#         val_dir = os.path.join(data_root, val_folder)

#         print("Olivetti train folder:", train_dir)
#         print("Olivetti validation folder:", val_dir)

#         if not os.path.exists(train_dir):
#             raise FileNotFoundError(f"Training folder not found: {train_dir}")

#         if not os.path.exists(val_dir):
#             raise FileNotFoundError(f"Validation folder not found: {val_dir}")

#         transform = transforms.Compose([
#             transforms.Grayscale(num_output_channels=1),
#             transforms.Resize((16, 16)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         train_data = ImageFolder(
#             root=train_dir,
#             transform=transform
#         )

#         test_data = ImageFolder(
#             root=val_dir,
#             transform=transform
#         )

#         if train_data.class_to_idx != test_data.class_to_idx:
#             raise ValueError(
#                 "Class mapping mismatch between training and validation folders. "
#                 "Please make sure both folders contain the same class structure."
#             )

#         print("Olivetti class mapping:", train_data.class_to_idx)
#         print("Training samples:", len(train_data))
#         print("Validation samples:", len(test_data))

#     else:
#         raise ValueError(f"Unsupported dataset: {dataset_name}")

#     # ============================================================
#     # Image augmentation only for image datasets
#     # ============================================================

#     if hyperparameters.get("augmentation", False) and dataset_name != "GATE_DRIVER":
#         augmented_transform = transforms.Compose([
#             transforms.RandomRotation(degrees=hyperparameters.get("rotation1", 10)),

#             transforms.RandomAffine(
#                 degrees=hyperparameters.get("rotation2", 10),
#                 translate=(0.1, 0.1),
#                 scale=(0.9, 1.1)
#             ),

#             transforms.RandomApply([
#                 transforms.ElasticTransform(alpha=40.0, sigma=4.0)
#             ], p=hyperparameters.get("elastictransformprobability", 0.0)),

#             transforms.Resize((16, 16)),

#             transforms.Grayscale(num_output_channels=1)
#             if dataset_name == "OLIVETTI"
#             else transforms.Lambda(lambda x: x),

#             transforms.ToTensor(),
#             transforms.Normalize(mean, std)
#         ])

#         if dataset_name == "OLIVETTI":
#             augmented_train_data = ImageFolder(
#                 root=train_dir,
#                 transform=augmented_transform
#             )
#         else:
#             augmented_train_data = base_dataset_train(
#                 root="data",
#                 transform=augmented_transform,
#                 download=True,
#                 **dataset_kwargs
#             )

#         train_data = ConcatDataset([train_data, augmented_train_data])

#     if hyperparameters.get("augmentation", False) and dataset_name == "GATE_DRIVER":
#         print("Warning: augmentation ignored for GATE_DRIVER tabular Excel dataset.")
#         hyperparameters["augmentation"] = False

#     # ============================================================
#     # Build model
#     # ============================================================

#     hyperparameters["num_classes"] = num_classes

#     model_params = {**hyperparameters, "num_classes": num_classes}

#     if dataset_name == "GATE_DRIVER":
#         model_params["input_dim"] = input_dim

#     model = load_model(
#         hyperparameters["model"],
#         model_params
#     )

#     model = model.to(device)

#     # ============================================================
#     # Check output size
#     # ============================================================

#     if dataset_name == "GATE_DRIVER":
#         dummy_input = torch.randn(1, input_dim).to(device)
#     else:
#         dummy_input = torch.randn(1, 1, 16, 16).to(device)

#     with torch.no_grad():
#         dummy_output = model(dummy_input)

#     print("Model output shape:", dummy_output.shape)
#     print("Expected output classes:", num_classes)

#     if dummy_output.shape[1] != num_classes:
#         raise ValueError(
#             f"Model output class mismatch. "
#             f"Model outputs {dummy_output.shape[1]} classes, "
#             f"but dataset needs {num_classes} classes."
#         )

#     if summary is not None:
#         try:
#             if dataset_name == "GATE_DRIVER":
#                 summary(model, input_size=(input_dim,))
#             else:
#                 summary(model, input_size=(1, 16, 16))
#         except Exception as e:
#             print("torchsummary skipped:", e)

#     print("training...")
#     train_model(model, device, hyperparameters, train_data, test_data)

#     print("saving model...")
#     os.makedirs("modeldata", exist_ok=True)
#     torch.save(model.state_dict(), f"modeldata/{runname}.pth")

#     print("Model saved to:", f"modeldata/{runname}.pth")

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

    if dataset_name != "GATE_DRIVER":
        raise ValueError("This training.py is prepared for dataset: GATE_DRIVER only.")

    train_data, val_data, num_classes, input_dim, feature_cols = load_gate_driver_excel(
        train_file=hyperparameters["train_file"],
        val_file=hyperparameters["val_file"],
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

    if summary is not None:
        try:
            if hyperparameters["model"] == "CNNMNIST":
                summary(model, input_size=(1, 16, 16))
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
