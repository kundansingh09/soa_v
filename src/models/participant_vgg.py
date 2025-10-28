"""
Defines the participant's model for Vertical Split Learning.

As per the paper, this is a VGG16 network that processes
a "half feature vector" (a split image) and outputs an embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class ParticipantVGG16(nn.Module):
    """
    Implements the VGG16-based feature extractor for a participant.
    
    It loads the VGG16 'features' module and modifies the
    final pooling layer to produce a fixed-size embedding.
    """
    def __init__(self, embedding_dim=512):
        super(ParticipantVGG16, self).__init__()
        
        # 1. Load the VGG16 feature extractor
        # We don't use the classifier head.
        vgg_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        
        # 2. Modify the model
        # The VGG16 'features' module is a nn.Sequential list.
        # The standard nn.Conv2d(3, 64, ...) input layer already
        # works with the (3, 32, 16) half-image.
        # We only modify the *end* of the network to produce the
        # desired embedding.
        
        # Replace the final MaxPool2d (at index 30) with
        # AdaptiveAvgPool2d to guarantee a fixed (1,1) spatial
        # output, which gives us a (B, 512, 1, 1) tensor.
        self.features = nn.Sequential(*list(vgg_features.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Define the "embedding" dimension
        # This will be the output of the AdaptiveAvgPool, flattened.
        self.embedding_dim = embedding_dim # 512 for VGG16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the split image 'x' through the VGG features
        to produce an embedding.
        
        Args:
            x: A batch of half-images, e.g., shape (B, 3, 32, 16).
        
        Returns:
            A batch of embedding vectors, shape (B, 512).
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        # Flatten (B, 512, 1, 1) -> (B, 512)
        embedding = torch.flatten(x, 1)
        return embedding

# --- Test Block ---
if __name__ == '__main__':
    # 1. Load config for input shape
    try:
        from config.cifar10 import SPLIT_METHOD, BATCH_SIZE
    except ImportError:
        print("Could not load config, using defaults.")
        SPLIT_METHOD = 'vertical'
        BATCH_SIZE = 4

    # 2. Determine dummy input shape
    if SPLIT_METHOD == 'vertical':
        dummy_input_shape = (BATCH_SIZE, 3, 32, 16)
    elif SPLIT_METHOD == 'horizontal':
        dummy_input_shape = (BATCH_SIZE, 3, 16, 32)
    else:
        raise ValueError(f"Invalid split method: {SPLIT_METHOD}")

    print(f"--- Testing ParticipantVGG16 ---")
    print(f"Dummy input shape (B, C, H, W): {dummy_input_shape}")
    
    # 3. Initialize model
    model = ParticipantVGG16()
    print(f"Model embedding dim: {model.embedding_dim}")
    # print(model) # Uncomment to see model architecture

    # 4. Test forward pass
    dummy_input = torch.randn(dummy_input_shape)
    output_embedding = model(dummy_input)
    
    print(f"Output embedding shape: {output_embedding.shape}")
    
    # 5. Verification
    expected_shape = (BATCH_SIZE, model.embedding_dim)
    assert output_embedding.shape == expected_shape
    print(f"\nSUCCESS: Output shape {output_embedding.shape} matches expected {expected_shape}.")