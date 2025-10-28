"""
Defines the participant's model for Vertical Split Learning.

Allows selection between VGG16 (original) and ResNet18 (new).
Models process a "half feature vector" (a split image) and output an embedding.
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights, resnet18, ResNet18_Weights

class ParticipantVGG16(nn.Module):
    """
    Implements the VGG16-based feature extractor for a participant.
    (Original implementation from previous step)
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        vgg_features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg_features.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = embedding_dim # 512 for VGG16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        embedding = torch.flatten(x, 1)
        return embedding

class ParticipantResNet18(nn.Module):
    """
    Implements the ResNet18-based feature extractor for a participant.
    
    Loads a pre-trained ResNet18, modifies the initial conv layer
    for CIFAR-sized inputs, removes the final FC layer, and ensures
    a consistent embedding output dimension.
    """
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # 1. Load pre-trained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 2. Modify conv1 for CIFAR-10 input size (and split)
        # Standard ResNet conv1 is kernel=7, stride=2, padding=3
        # Standard CIFAR modification is kernel=3, stride=1, padding=1
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        
        # Remove the initial MaxPool layer typically used after conv1
        # for CIFAR-10 adaptation, as our input is already small.
        resnet.maxpool = nn.Identity()
        
        # 3. Remove the final classification layer (fc)
        # We'll use all layers *before* the fc layer as the feature extractor
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)
        
        # 4. Define the embedding dimension
        # ResNet18's output before fc is 512
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes the split image 'x' through the ResNet features
        to produce an embedding.
        
        Args:
            x: A batch of half-images, e.g., shape (B, 3, 32, 16).
        
        Returns:
            A batch of embedding vectors, shape (B, 512).
        """
        x = self.feature_extractor(x)
        # Flatten the output: (B, 512, 1, 1) -> (B, 512)
        embedding = torch.flatten(x, 1)
        return embedding

# --- Factory Function ---
def get_participant_model(model_type: str) -> nn.Module:
    """
    Returns an instance of the specified participant model.
    """
    if model_type == 'vgg16':
        return ParticipantVGG16()
    elif model_type == 'resnet18':
        return ParticipantResNet18()
    else:
        raise ValueError(f"Unknown participant model type: {model_type}")

# --- Test Block ---
if __name__ == '__main__':
    # Test ResNet18
    try:
        from config.cifar10 import SPLIT_METHOD, BATCH_SIZE
    except ImportError:
        SPLIT_METHOD = 'vertical'
        BATCH_SIZE = 4

    if SPLIT_METHOD == 'vertical':
        dummy_input_shape = (BATCH_SIZE, 3, 32, 16)
    else: # horizontal
        dummy_input_shape = (BATCH_SIZE, 3, 16, 32)

    print(f"--- Testing ParticipantResNet18 ---")
    print(f"Dummy input shape (B, C, H, W): {dummy_input_shape}")
    
    model = get_participant_model('resnet18')
    print(f"Model embedding dim: {model.embedding_dim}")

    dummy_input = torch.randn(dummy_input_shape)
    output_embedding = model(dummy_input)
    print(f"Output embedding shape: {output_embedding.shape}")
    
    expected_shape = (BATCH_SIZE, model.embedding_dim)
    assert output_embedding.shape == expected_shape
    print(f"\nSUCCESS: ResNet18 Output shape {output_embedding.shape} matches expected {expected_shape}.")

    # Test VGG16 (Optional)
    print(f"\n--- Testing ParticipantVGG16 ---")
    model_vgg = get_participant_model('vgg16')
    output_vgg = model_vgg(dummy_input)
    assert output_vgg.shape == (BATCH_SIZE, model_vgg.embedding_dim)
    print("SUCCESS: VGG16 Test Passed.")