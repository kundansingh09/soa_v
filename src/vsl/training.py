"""
Implements the benign training logic for Vertical Split Learning (VSL).

This module contains the function for training the models for one
epoch, following the 3-step VSL process from the VILLAIN paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from typing import Tuple, List

# --- Main Training Function ---

def train_benign_epoch(
    model_A: nn.Module, optimizer_A: optim.Optimizer,
    model_B: nn.Module, optimizer_B: optim.Optimizer,
    model_S: nn.Module, optimizer_S: optim.Optimizer,
    loader_A: DataLoader,
    loader_B: DataLoader,
    loader_S: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Trains all models for one benign epoch using VSL.

    Args:
        model_A, model_B, model_S: The participant and server models.
        optimizer_A, optimizer_B, optimizer_S: The optimizers.
        loader_A, loader_B, loader_S: The synchronized data loaders.
        criterion: The loss function (e.g., CrossEntropyLoss).
        device: The device to run on (e.g., 'cuda' or 'cpu').

    Returns:
        A tuple of (average_epoch_loss, epoch_accuracy).
    """
    # Set all models to training mode
    model_A.train()
    model_B.train()
    model_S.train()

    running_loss = 0.0
    correct_preds = 0
    total_samples = 0

    # Use zip to iterate over synchronized data loaders
    # The loaders yield (data_A, data_B, labels_S)
    batch_iterator = zip(loader_A, loader_B, loader_S)
    
    for i, (data_A, data_B, labels_S) in enumerate(batch_iterator):
        # Move data to the specified device
        data_A, data_B = data_A.to(device), data_B.to(device)
        labels_S = labels_S.to(device)

        # --- 1. Zero all gradients ---
        optimizer_A.zero_grad()
        optimizer_B.zero_grad()
        optimizer_S.zero_grad()

        # --- 2. Step I: Participants' Forward Pass [cite: 115-119] ---
        # Compute embeddings
        e_A = model_A(data_A)
        e_B = model_B(data_B)

        # --- 3. Step II: Server's Forward & Backward Pass [cite: 120-122] ---
        # To compute gradients *only* for the server, we detach
        # embeddings from the participants' computation graphs.
        # We explicitly tell PyTorch to track gradients on these
        # new detached tensors.
        e_A_server = e_A.detach().requires_grad_()
        e_B_server = e_B.detach().requires_grad_()

        # Aggregate embeddings (Concatenation)
        e_agg = torch.cat((e_A_server, e_B_server), dim=1)
        
        # Server computes logits and loss
        logits = model_S(e_agg)
        loss = criterion(logits, labels_S)

        # Server backward pass (computes \nabla L)
        loss.backward()

        # Server optimizer updates its model (f^s)
        optimizer_S.step()

        # --- 4. Step III: Participants' Backward Pass [cite: 123-124] ---
        # The server "sends back" the gradients (g_k = \nabla_{e^k} L)
        # which are now stored in .grad of the detached tensors.
        g_A = e_A_server.grad
        g_B = e_B_server.grad
        
        # Participants perform their backward pass using *only*
        # the gradients from the server.
        # This backpropagates g_A and g_B through f^k.
        e_A.backward(g_A)
        e_B.backward(g_B)

        # Participants' optimizers update their models (f^k)
        optimizer_A.step()
        optimizer_B.step()

        # --- 5. Update Statistics ---
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total_samples += labels_S.size(0)
        correct_preds += (predicted == labels_S).sum().item()

    # --- 6. Calculate Epoch Statistics ---
    avg_epoch_loss = running_loss / len(loader_A)
    epoch_accuracy = 100.0 * correct_preds / total_samples

    return avg_epoch_loss, epoch_accuracy

# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing VSL Benign Training Epoch ---")

    # Add project root to sys.path
    project_root = torch.tensor(0.0).device.type # A bit of a hack to get project root
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from config.cifar10 import (
            BATCH_SIZE, SPLIT_METHOD, BENIGN_LR, 
            SERVER_LR, PARTICIPANT_MODEL_TYPE
        )
        from data.cifar10_loader import load_cifar10, get_vsl_dataloaders
        from src.models.participant_vgg import ParticipantVGG16
        from src.models.server_model import ServerModel
    except ImportError as e:
        print(f"Error: Could not import necessary modules. {e}")
        print("Please ensure config, data, and model files exist.")
        sys.exit(1)

    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # 2. Load one batch of data
    print("Loading one batch of CIFAR-10 data...")
    trainset, _ = load_cifar10()
    # Use a small subset for a quick test
    mini_trainset = torch.utils.data.Subset(trainset, range(BATCH_SIZE * 2))
    
    loader_A, loader_B, loader_S = get_vsl_dataloaders(
        original_dataset=mini_trainset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=True
    )

    # 3. Initialize Models
    print("Initializing models...")
    model_A = ParticipantVGG16().to(DEVICE)
    model_B = ParticipantVGG16().to(DEVICE)
    model_S = ServerModel(
        participant_embedding_dim=model_A.embedding_dim,
        num_participants=2,
        num_classes=10
    ).to(DEVICE)

    # 4. Initialize Optimizers and Criterion
    optimizer_A = optim.Adam(model_A.parameters(), lr=BENIGN_LR)
    optimizer_B = optim.Adam(model_B.parameters(), lr=BENIGN_LR)
    optimizer_S = optim.Adam(model_S.parameters(), lr=SERVER_LR)
    criterion = nn.CrossEntropyLoss()

    # 5. Run one test epoch
    print("Running one benign training epoch...")
    try:
        loss, acc = train_benign_epoch(
            model_A, optimizer_A, model_B, optimizer_B, model_S, optimizer_S,
            loader_A, loader_B, loader_S, criterion, DEVICE
        )
        print(f"\n--- Test Epoch Complete ---")
        print(f"Average Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.2f}%")
        print("\nSUCCESS: Benign epoch ran without errors.")
    
    except Exception as e:
        print(f"\nFAILURE: Error during test epoch: {e}")
        import traceback
        traceback.print_exc()