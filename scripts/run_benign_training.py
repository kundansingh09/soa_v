"""
Main script to run a benign training process for VSL on CIFAR-10.

This script:
1. Loads the configuration.
2. Loads the CIFAR-10 dataset and VSL data loaders.
3. Initializes the two participant models (VGG16) and the server
   model (3-layer FC).
4. Runs the benign training loop for a specified number of epochs.
5. Saves the final trained model weights to 'results/models/'.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from typing import Dict

# --- 1. Add Project Root to sys.path ---
# This ensures imports from config, data, src, etc., work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. Project-specific Imports ---
try:
    from config.cifar10 import (
        BATCH_SIZE, SPLIT_METHOD, BENIGN_LR, SERVER_LR, 
        PARTICIPANT_MODEL_TYPE, NUM_PARTICIPANTS, EMBEDDING_AGGREGATION
    )
    from data.cifar10_loader import load_cifar10, get_vsl_dataloaders
    from src.models.participant_vgg import ParticipantVGG16
    from src.models.server_model import ServerModel
    from src.vsl.training import train_benign_epoch
except ImportError as e:
    print(f"Error: Could not import necessary modules.")
    print(f"Details: {e}")
    print(f"Please ensure all modules are correctly defined and sys.path is correct.")
    sys.exit(1)

# --- 3. Configuration ---
# The paper does not specify the number of epochs. We assume a
# reasonable number for CIFAR-10 convergence.
NUM_EPOCHS = 50
MODEL_SAVE_DIR = os.path.join(project_root, "results", "models")
BENIGN_A_PATH = os.path.join(MODEL_SAVE_DIR, "benign_model_A.pth")
BENIGN_B_PATH = os.path.join(MODEL_SAVE_DIR, "benign_model_B.pth")
BENIGN_S_PATH = os.path.join(MODEL_SAVE_DIR, "benign_model_S.pth")


def main():
    """
    Main function to orchestrate the benign training process.
    """
    
    # --- 1. Setup ---
    print("--- 🚀 Starting Benign VSL Training ---")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    DEVICE = torch.device("cuda" if torch.cuda.is.available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Total Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Split Method: {SPLIT_METHOD}")

    # --- 2. Load Data ---
    print("\nLoading CIFAR-10 dataset...")
    trainset, testset = load_cifar10()
    
    train_loader_A, train_loader_B, train_loader_S = get_vsl_dataloaders(
        original_dataset=trainset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=True
    )
    print("Data loaders created.")
    
    # --- 3. Initialize Models ---
    print("Initializing models...")
    if PARTICIPANT_MODEL_TYPE != 'vgg16':
        raise ValueError("This script is configured for VGG16 participant models.")
        
    model_A = ParticipantVGG16().to(DEVICE)
    model_B = ParticipantVGG16().to(DEVICE)
    
    if EMBEDDING_AGGREGATION != 'concatenation':
        raise ValueError("This script assumes 'concatenation' aggregation.")
        
    model_S = ServerModel(
        participant_embedding_dim=model_A.embedding_dim,
        num_participants=NUM_PARTICIPANTS,
        num_classes=10
    ).to(DEVICE)

    # --- 4. Initialize Optimizers & Criterion ---
    optimizer_A = optim.Adam(model_A.parameters(), lr=BENIGN_LR)
    optimizer_B = optim.Adam(model_B.parameters(), lr=BENIGN_LR)
    optimizer_S = optim.Adam(model_S.parameters(), lr=SERVER_LR)
    
    criterion = nn.CrossEntropyLoss()
    print("Models and optimizers initialized.")

    # --- 5. Training Loop ---
    print("\n--- Starting Training Loop ---")
    for epoch in range(NUM_EPOCHS):
        try:
            loss, acc = train_benign_epoch(
                model_A, optimizer_A, model_B, optimizer_B, model_S, optimizer_S,
                train_loader_A, train_loader_B, train_loader_S, 
                criterion, DEVICE
            )
            
            print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] | "
                  f"Loss: {loss:.4f} | "
                  f"Train Accuracy: {acc:.2f}%")
                  
        except Exception as e:
            print(f"\nError during training epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            print("Stopping training.")
            break

    # --- 6. Save Final Models ---
    print("\n--- Training Complete ---")
    try:
        torch.save(model_A.state_dict(), BENIGN_A_PATH)
        torch.save(model_B.state_dict(), BENIGN_B_PATH)
        torch.save(model_S.state_dict(), BENIGN_S_PATH)
        print(f"Benign models saved to {MODEL_SAVE_DIR}")
    except Exception as e:
        print(f"Error saving models: {e}")
        
    print("--- ✅ Benign Training Script Finished ---")


if __name__ == "__main__":
    main()