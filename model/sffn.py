import torch
import torch.nn as nn
import timm

class FrequencyCNN(nn.Module):
    """A simple 3-layer CNN for the frequency stream."""
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32); self.relu = nn.ReLU(inplace=True); self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 128

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return self.global_pool(x)

class SFFN(nn.Module):
    """Spatial-Frequency Fusion Network (SFFN) with EfficientNet backbone"""
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.spatial_stream = timm.create_model('efficientnet_lite0', pretrained=pretrained)
        spatial_feature_dim = self.spatial_stream.classifier.in_features
        self.spatial_stream.classifier = nn.Identity()
        self.frequency_stream = FrequencyCNN(in_channels=2)
        freq_feature_dim = self.frequency_stream.feature_dim
        fusion_dim = spatial_feature_dim + freq_feature_dim
        self.fusion_mlp = nn.Sequential(nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))
        self.reconstruction_head = nn.Sequential(
            nn.ConvTranspose2d(freq_feature_dim, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1))

    def forward(self, x_spatial, x_freq):
        spatial_features = self.spatial_stream(x_spatial)
        freq_features_map = self.frequency_stream(x_freq)
        freq_features = freq_features_map.view(freq_features_map.size(0), -1)
        fused_features = torch.cat((spatial_features, freq_features), dim=1)
        classification_output = self.fusion_mlp(fused_features)
        reconstruction_output = self.reconstruction_head(freq_features_map)
        return classification_output, reconstruction_output