# meso_inception4.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MesoInception4(nn.Module):
    """
    MesoInception-4 model architecture for deepfake detection.
    
    This model is a compact CNN designed for real-time inference on mesoscopic
    image features. The Inception blocks capture features at multiple
    scales efficiently, which is key for detecting subtle manipulation
    artifacts introduced during forgery.
    
    Args:
        num_classes (int): The number of output classes. 
                           Use 1 for binary classification (deepfake detection)
                           and 10 for CIFAR-10 classification.
    """
    def __init__(self, num_classes=10):
        super(MesoInception4, self).__init__()
        self.num_classes = num_classes
        
        # Inception Block 1: Captures multi-scale features from RGB input.
        # Input shape: (B, 3, 256, 256)
        # 1x1 conv branch
        self.inception1_conv1_1x1 = nn.Conv2d(3, 1, kernel_size=1, padding=0, bias=False)
        
        # 1x1 followed by 3x3 conv branch
        self.inception1_conv2_1x1 = nn.Conv2d(3, 4, kernel_size=1, padding=0, bias=False)
        self.inception1_conv2_3x3 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        
        # 1x1 followed by 3x3 dilated conv (dilation=2) branch
        self.inception1_conv3_1x1 = nn.Conv2d(3, 4, kernel_size=1, padding=0, bias=False)
        self.inception1_conv3_3x3_dil2 = nn.Conv2d(4, 4, kernel_size=3, padding=2, dilation=2, bias=False)
        
        # 1x1 followed by 3x3 dilated conv (dilation=3) branch
        self.inception1_conv4_1x1 = nn.Conv2d(3, 2, kernel_size=1, padding=0, bias=False)
        self.inception1_conv4_3x3_dil3 = nn.Conv2d(2, 2, kernel_size=3, padding=3, dilation=3, bias=False)
        
        self.inception1_bn = nn.BatchNorm2d(1 + 4 + 4 + 2)
        
        # Inception Block 2: Further refines multi-scale features from previous block.
        # Input shape: (B, 11, 128, 128)
        # 1x1 conv branch
        self.inception2_conv1_1x1 = nn.Conv2d(11, 2, kernel_size=1, padding=0, bias=False)
        
        # 1x1 followed by 3x3 conv branch
        self.inception2_conv2_1x1 = nn.Conv2d(11, 4, kernel_size=1, padding=0, bias=False)
        self.inception2_conv2_3x3 = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False)
        
        # 1x1 followed by 3x3 dilated conv (dilation=2) branch
        self.inception2_conv3_1x1 = nn.Conv2d(11, 4, kernel_size=1, padding=0, bias=False)
        self.inception2_conv3_3x3_dil2 = nn.Conv2d(4, 4, kernel_size=3, padding=2, dilation=2, bias=False)
        
        # 1x1 followed by 3x3 dilated conv (dilation=3) branch
        self.inception2_conv4_1x1 = nn.Conv2d(11, 2, kernel_size=1, padding=0, bias=False)
        self.inception2_conv4_3x3_dil3 = nn.Conv2d(2, 2, kernel_size=3, padding=3, dilation=3, bias=False)
        
        self.inception2_bn = nn.BatchNorm2d(2 + 4 + 4 + 2)
        
        # Standard Convolutional Layers
        self.conv1 = nn.Conv2d(12, 16, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        # Classifier
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, self.num_classes)
        
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # Inception Block 1
        x1_branch1 = self.inception1_conv1_1x1(x)
        x1_branch2 = self.inception1_conv2_1x1(x)
        x1_branch2 = self.inception1_conv2_3x3(x1_branch2)
        x1_branch3 = self.inception1_conv3_1x1(x)
        x1_branch3 = self.inception1_conv3_3x3_dil2(x1_branch3)
        x1_branch4 = self.inception1_conv4_1x1(x)
        x1_branch4 = self.inception1_conv4_3x3_dil3(x1_branch4)
        x = torch.cat([x1_branch1, x1_branch2, x1_branch3, x1_branch4], 1)
        x = self.inception1_bn(x)
        x = self.leakyrelu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Inception Block 2
        x2_branch1 = self.inception2_conv1_1x1(x)
        x2_branch2 = self.inception2_conv2_1x1(x)
        x2_branch2 = self.inception2_conv2_3x3(x2_branch2)
        x2_branch3 = self.inception2_conv3_1x1(x)
        x2_branch3 = self.inception2_conv3_3x3_dil2(x2_branch3)
        x2_branch4 = self.inception2_conv4_1x1(x)
        x2_branch4 = self.inception2_conv4_3x3_dil3(x2_branch4)
        x = torch.cat([x2_branch1, x2_branch2, x2_branch3, x2_branch4], 1)
        x = self.inception2_bn(x)
        x = self.leakyrelu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Standard Convolutional Blocks
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.leakyrelu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=4)
        
        # Flatten and Classify
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.leakyrelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x