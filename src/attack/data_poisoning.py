"""
Implements Module 2 of the VILLAIN attack: Data Poisoning.

This module contains the functions for:
1.  Creating the trigger mask based on feature variance.
2.  Fabricating the additive trigger $\mathcal{E}$.
3.  Augmenting the trigger with Dropout and Shifting.
"""

import torch
import random
from typing import List

def create_trigger_mask(embeddings: torch.Tensor, m: int) -> torch.Tensor:
    """
    Finds the m elements in the embedding vector with the
    highest standard deviation across all samples.

    Args:
        embeddings: A 2D tensor of (N_samples, embed_dim)
                    used to calculate variance.
        m: The number of elements to select for the trigger mask.

    Returns:
        A 1D binary tensor of shape (embed_dim) where 1s
        indicate the trigger area.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"Embeddings must be 2D, but got {embeddings.dim()}D")
    if m > embeddings.shape[1]:
        raise ValueError(f"m ({m}) cannot be larger than embed_dim ({embeddings.shape[1]})")

    # Calculate standard deviation for each element (column)
    # across all samples (rows)
    std_dev_per_element = torch.std(embeddings, dim=0)
    
    # Get the indices of the m largest std dev values
    _, top_m_indices = torch.topk(std_dev_per_element, k=m)
    
    # Create the mask
    mask = torch.zeros_like(std_dev_per_element)
    mask[top_m_indices] = 1.0
    
    return mask

def fabricate_trigger(mask: torch.Tensor,
                      std_dev_all: torch.Tensor,
                      beta: float) -> torch.Tensor:
    """
    Creates the additive trigger $\mathcal{E}$ based on the mask
    and the specified pattern.
    
    The formula is: $\mathcal{E} = \mathcal{M} \otimes (\beta \cdot \Delta)$
    where $\Delta = [+\delta, +\delta, -\delta, -\delta, ...]$
    and $\delta$ is the average std dev of the masked elements.

    Args:
        mask: The 1D binary trigger mask (from create_trigger_mask).
        std_dev_all: A 1D tensor of std dev for all elements.
        beta: The trigger magnitude (β).

    Returns:
        The fabricated trigger $\mathcal{E}$ (a 1D tensor).
    """
    if mask.dim() != 1 or std_dev_all.dim() != 1:
        raise ValueError("Mask and std_dev_all must be 1D tensors.")
    
    # Get delta (δ): "the average standard deviation of elements in
    # the backdoor dimension of all samples"
    masked_std_devs = std_dev_all[mask == 1.0]
    if len(masked_std_devs) == 0:
        return torch.zeros_like(mask) # Avoid division by zero if mask is empty
        
    delta = masked_std_devs.mean().item()

    # Create the repeating pattern [+δ, +δ, -δ, -δ]
    embed_dim = mask.shape[0]
    pattern = torch.tensor(
        [delta, delta, -delta, -delta], 
        dtype=std_dev_all.dtype, 
        device=std_dev_all.device
    )
    
    # Tile the pattern to fill the embedding dimension
    delta_pattern = pattern.repeat(embed_dim // 4 + 1)[:embed_dim]

    # Calculate the trigger: $\mathcal{E} = \mathcal{M} \otimes (\beta \cdot \Delta)$
    trigger = mask * (beta * delta_pattern)
    
    return trigger


def augment_trigger(trigger: torch.Tensor,
                    mask: torch.Tensor,
                    dropout_ratio: float,
                    shifting_range: List[float]) -> torch.Tensor:
    """
    Applies Dropout and Shifting to the fabricated trigger.

    Args:
        trigger: The original trigger $\mathcal{E}$ (1D tensor).
        mask: The original binary mask (1D tensor).
        dropout_ratio: The fraction of 1s in the mask to
                       randomly set to 0.
        shifting_range: A [min, max] list for the uniform
                        random shift multiplier.

    Returns:
        The augmented trigger (a 1D tensor).
    """
    
    # --- 1. Apply Dropout ---
    aug_mask = mask.clone()
    
    # Get the indices where the trigger is active
    trigger_indices = aug_mask.nonzero().squeeze()
    
    if trigger_indices.numel() > 0:
        # Calculate how many trigger elements to drop
        num_to_drop = int(trigger_indices.numel() * dropout_ratio)
        
        # Randomly select indices to drop
        drop_indices = trigger_indices[
            torch.randperm(trigger_indices.numel())[:num_to_drop]
        ]
        
        # Set the dropped indices to 0 in the augmented mask
        aug_mask[drop_indices] = 0.0

    # --- 2. Apply Shifting ---
    # Get random shift value γ
    shift_val = random.uniform(shifting_range[0], shifting_range[1])
    
    # Apply shift to the *original* trigger
    shifted_trigger = trigger * shift_val
    
    # --- 3. Combine ---
    # Apply the dropped-out mask to the shifted trigger
    augmented_trigger = shifted_trigger * aug_mask
    
    return augmented_trigger

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing Data Poisoning Module ---")
    
    EMBED_DIM = 512
    N_SAMPLES = 200
    M = 50  # 50 elements in the mask
    BETA = 0.4
    DROPOUT_RATIO = 0.75
    SHIFT_RANGE = [0.6, 1.2]

    # 1. Create dummy embedding data
    # Make some columns high-variance, others low-variance
    dummy_embeds = torch.randn(N_SAMPLES, EMBED_DIM)
    dummy_embeds[:, :M] *= 10.0 # First M elements have high std dev
    
    print(f"Dummy data shape: {dummy_embeds.shape}")

    # 2. Test create_trigger_mask
    mask = create_trigger_mask(dummy_embeds, m=M)
    print(f"\nMask shape: {mask.shape}")
    print(f"Mask active elements: {mask.sum().item()} (Expected: {M})")
    assert mask.sum().item() == M
    # Check that it picked the first M elements
    assert mask[:M].sum().item() == M 
    print("SUCCESS: create_trigger_mask")

    # 3. Test fabricate_trigger
    std_all = torch.std(dummy_embeds, dim=0)
    trigger = fabricate_trigger(mask, std_all, beta=BETA)
    print(f"\nTrigger shape: {trigger.shape}")
    print(f"Trigger non-zero elements: {torch.count_nonzero(trigger).item()} (Expected: {M})")
    assert torch.count_nonzero(trigger).item() == M
    # Check that only masked elements are non-zero
    assert torch.all((trigger == 0.0) == (mask == 0.0))
    print("SUCCESS: fabricate_trigger")

    # 4. Test augment_trigger
    aug_trigger = augment_trigger(
        trigger, 
        mask, 
        dropout_ratio=DROPOUT_RATIO, 
        shifting_range=SHIFT_RANGE
    )
    print(f"\nAugmented Trigger shape: {aug_trigger.shape}")
    
    num_non_zero_aug = torch.count_nonzero(aug_trigger).item()
    expected_non_zero = int(M * (1.0 - DROPOUT_RATIO))
    print(f"Augmented non-zero elements: {num_non_zero_aug} (Expected: ~{expected_non_zero})")
    
    # Check that magnitude is shifted
    original_mag = torch.sum(torch.abs(trigger))
    aug_mag = torch.sum(torch.abs(aug_trigger))
    print(f"Original Mag Sum: {original_mag.item():.2f}")
    print(f"Augmented Mag Sum: {aug_mag.item():.2f}")
    
    assert num_non_zero_aug <= M
    assert not torch.allclose(original_mag, aug_mag)
    print("SUCCESS: augment_trigger")