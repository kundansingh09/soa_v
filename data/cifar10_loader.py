"""
Loads and preprocesses the CIFAR-10 dataset for Vertical Split Learning (VSL).

This module downloads CIFAR-10 and splits each image into two
"half feature vectors" for the two participants, as described
in the VILLAIN paper's experimental setup.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset

# --- 1. Custom Dataset for VSL ---

class _SplitCIFAR10Dataset(Dataset):
    """
    A custom wrapper for the CIFAR-10 dataset to facilitate
    Vertical Split Learning. Based on the participant_id,
    it returns either a split half of the image or the label.
    """
    def __init__(self, 
                 original_dataset: Dataset, 
                 split_method: str, 
                 participant_id: str):
        """
        Args:
            original_dataset: The torchvision CIFAR-10 dataset.
            split_method: How to split the image ('vertical' or 'horizontal').
            participant_id: Who is accessing the data ('A', 'B', or 'Server').
        """
        self.original_dataset = original_dataset
        self.split_method = split_method
        self.participant_id = participant_id

        if self.split_method not in ['vertical', 'horizontal']:
            raise ValueError(f"Unknown split_method: {self.split_method}")
        if self.participant_id not in ['A', 'B', 'Server']:
            raise ValueError(f"Unknown participant_id: {self.participant_id}")

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        """
        Fetches the data for the given index.
        - Server gets the label.
        - Participants A and B get their respective image halves.
        """
        # Get the original image and label
        image, label = self.original_dataset[idx] # original_dataset IS the Subset here if shuffled

        # --- Server ---
        if self.participant_id == 'Server':
            # Return BOTH label and the original index 'idx'
            return label, idx

        # --- Participants A and B ---
        if self.split_method == 'vertical':
            # Split along the Width (W) dimension (dim=2)
            # (3, 32, 32) -> two (3, 32, 16) tensors
            half1, half2 = torch.split(image, 16, dim=2)
        
        elif self.split_method == 'horizontal':
            # Split along the Height (H) dimension (dim=1)
            # (3, 32, 32) -> two (3, 16, 32) tensors
            half1, half2 = torch.split(image, 16, dim=1)

        if self.participant_id == 'A':
            return half1
        else: # self.participant_id == 'B'
            return half2

# --- 2. Public Functions ---

def load_cifar10(data_root: str = './data_cache'):
    """
    Downloads the CIFAR-10 dataset using torchvision.
    
    The paper does not specify transforms, so we use ToTensor()
    which is the minimum required to work with PyTorch models.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download training data
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    # Download test data
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    return trainset, testset

def get_vsl_dataloaders(original_dataset: Dataset,
                        batch_size: int,
                        split_method: str,
                        shuffle: bool = True,
                        num_workers: int = 2):
    """
    Creates three synchronized DataLoader objects: one for Participant A,
    one for Participant B, and one for the Server.

    This function ensures that all three loaders are synchronized by
    using a single set of shuffled indices (via torch.Subset)
    if shuffle=True.
    
    Args:
        original_dataset: The CIFAR-10 dataset (e.g., trainset or testset).
        batch_size: The batch size for the loaders.
        split_method: The method to split the image ('vertical' or 'horizontal').
        shuffle: Whether to shuffle the data.
        num_workers: Number of workers for the DataLoader.

    Returns:
        A tuple of (loader_A, loader_B, loader_Server)
    """
    
    # Create the three custom dataset views
    dataset_A = _SplitCIFAR10Dataset(original_dataset, split_method, 'A')
    dataset_B = _SplitCIFAR10Dataset(original_dataset, split_method, 'B')
    dataset_S = _SplitCIFAR10Dataset(original_dataset, split_method, 'Server')

    loader_A, loader_B, loader_S = None, None, None

    if shuffle:
        # 1. Generate one set of indices to shuffle the entire dataset
        indices = torch.randperm(len(original_dataset))
        
        # 2. Create Subsets for each participant using the *same* shuffled indices
        subset_A = Subset(dataset_A, indices)
        subset_B = Subset(dataset_B, indices)
        subset_S = Subset(dataset_S, indices)
        
        # 3. Create DataLoaders from the Subsets with shuffle=False
        #    (The data is already shuffled by the Subset)
        loader_A = DataLoader(subset_A, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loader_B = DataLoader(subset_B, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loader_S = DataLoader(subset_S, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        # No shuffle, create DataLoaders directly from the datasets
        loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loader_S = DataLoader(dataset_S, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader_A, loader_B, loader_S

# --- 3. Test Block ---

if __name__ == '__main__':
    print("Testing CIFAR-10 VSL Data Loader...")
    
    # 1. Load config
    try:
        from config.cifar10 import BATCH_SIZE, SPLIT_METHOD
    except ImportError:
        print("Could not load config, using defaults.")
        BATCH_SIZE = 4
        SPLIT_METHOD = 'vertical'

    # 2. Load dataset
    trainset, _ = load_cifar10()

    # 3. Get data loaders
    loader_A, loader_B, loader_S = get_vsl_dataloaders(
        original_dataset=trainset,
        batch_size=BATCH_SIZE,
        split_method=SPLIT_METHOD,
        shuffle=True
    )

    # 4. Iterate over one batch to check synchronization
    print(f"Batch Size: {BATCH_SIZE}, Split Method: {SPLIT_METHOD}")
    print("Fetching one batch from all three loaders...")

    try:
        data_A = next(iter(loader_A))
        data_B = next(iter(loader_B))
        data_S = next(iter(loader_S))

        print("\n--- Batch Shapes ---")
        print(f"Participant A data shape: {data_A.shape}")
        print(f"Participant B data shape: {data_B.shape}")
        print(f"Server (labels) shape:  {data_S.shape}")

        # 5. Verification
        if SPLIT_METHOD == 'vertical':
            # (B, C, H, W/2) -> (4, 3, 32, 16)
            assert data_A.shape == (BATCH_SIZE, 3, 32, 16)
            assert data_B.shape == (BATCH_SIZE, 3, 32, 16)
        elif SPLIT_METHOD == 'horizontal':
            # (B, C, H/2, W) -> (4, 3, 16, 32)
            assert data_A.shape == (BATCH_SIZE, 3, 16, 32)
            assert data_B.shape == (BATCH_SIZE, 3, 16, 32)
        
        assert data_S.shape == (BATCH_SIZE,)

        print("\nSUCCESS: Loaders are synchronized and shapes are correct.")

    except Exception as e:
        print(f"\nFAILURE: Error during loader test: {e}")