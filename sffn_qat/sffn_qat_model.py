import torch
import torch.nn as nn
import timm
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat
from brevitas.quant import Int16WeightPerTensorFloat, Int16ActPerTensorFloat
from brevitas.proxy import TimmModelProxy

# --- INT16 Quantizer Config ---
# We use per-tensor for weights here; for ConvTranspose, per-channel is less common.
class Int16Quant:
    Act = Int16ActPerTensorFloat
    Weight = Int16WeightPerTensorFloat
    Bias = Int16WeightPerTensorFloat # Or scale-invariant, e.g., Int32Bias

# --- INT8 Quantizer Config ---
class Int8Quant:
    Act = Int8ActPerTensorFloat
    Weight = Int8WeightPerChannelFloat # Per-channel is best practice for Conv/Linear
    Bias = Int16WeightPerTensorFloat # Biases are often kept in higher precision

class FrequencyCNN_QAT(nn.Module):
    """
    Brevitas-quantized 3-layer CNN (INT16).
    This module is refactored to be fusion-friendly (using nn.Sequential).
    """
    def __init__(self, in_channels=2):
        super().__init__()
        
        # Quantize the input to INT16
        self.quant_inp = qnn.QuantIdentity(
            act_quant=Int16Quant.Act, return_quant_tensor=True
        )
        
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(
                in_channels, 32, kernel_size=3, stride=1, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=True
            ),
            nn.BatchNorm2d(32),
            qnn.QuantReLU(
                act_quant=Int16Quant.Act, return_quant_tensor=True
            )
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(
                32, 64, kernel_size=3, stride=1, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=True
            ),
            nn.BatchNorm2d(64),
            qnn.QuantReLU(
                act_quant=Int16Quant.Act, return_quant_tensor=True
            )
        )
        
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(
                64, 128, kernel_size=3, stride=1, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=True
            ),
            nn.BatchNorm2d(128),
            qnn.QuantReLU(
                act_quant=Int16Quant.Act, return_quant_tensor=True
            )
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 128

    def forward(self, x):
        # x is FP32
        x = self.quant_inp(x) # x is now INT16
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = self.pool(self.layer3(x))
        # output is INT16
        return self.global_pool(x)


class SFFN_QAT(nn.Module):
    """
    Brevitas-quantized SFFN (Mixed-Precision INT8/INT16).
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # --- Spatial Stream (INT8) ---
        # We use Brevitas's TimmModelProxy to wrap EfficientNet-lite0
        # and apply INT8 quantization to it.
        self.spatial_stream = TimmModelProxy(
            'efficientnet_lite0',
            pretrained=pretrained,
            weight_quant=Int8Quant.Weight,
            act_quant=Int8Quant.Act,
            bias_quant=Int8Quant.Bias,
            return_quant_tensor=True
        )
        # Grab the feature dimension BEFORE replacing the classifier
        spatial_feature_dim = self.spatial_stream.model.classifier.in_features
        # Replace the classifier with an Identity
        self.spatial_stream.model.classifier = nn.Identity()
        
        # Quantize the input to the spatial stream
        self.quant_spatial_inp = qnn.QuantIdentity(
            act_quant=Int8Quant.Act, return_quant_tensor=True
        )

        # --- Frequency Stream (INT16) ---
        self.frequency_stream = FrequencyCNN_QAT(in_channels=2)
        freq_feature_dim = self.frequency_stream.feature_dim

        fusion_dim = spatial_feature_dim + freq_feature_dim

        # --- Fusion MLP (INT8) ---
        self.fusion_mlp = nn.Sequential(
            qnn.QuantLinear(
                fusion_dim, 512,
                weight_quant=Int8Quant.Weight,
                bias_quant=Int8Quant.Bias,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(act_quant=Int8Quant.Act, return_quant_tensor=True),
            nn.Dropout(0.5),
            qnn.QuantLinear(
                512, num_classes,
                weight_quant=Int8Quant.Weight,
                bias_quant=Int8Quant.Bias,
                return_quant_tensor=False # Output FP32
            )
        )
        
        # --- Reconstruction Head (INT16) ---
        self.reconstruction_head = nn.Sequential(
            qnn.QuantConvTranspose2d(
                freq_feature_dim, 64, kernel_size=4, stride=2, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(act_quant=Int16Quant.Act, return_quant_tensor=True),
            qnn.QuantConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=True
            ),
            qnn.QuantReLU(act_quant=Int16Quant.Act, return_quant_tensor=True),
            qnn.QuantConvTranspose2d(
                32, 2, kernel_size=3, stride=1, padding=1,
                weight_quant=Int16Quant.Weight,
                bias_quant=Int16Quant.Bias,
                return_quant_tensor=False # Output FP32
            )
        )
        
        # --- Dequantization Stubs for Fusion ---
        # These will dequantize the outputs of the two streams back to FP32
        # so they can be concatenated.
        self.dequant_spatial = qnn.QuantIdentity(return_quant_tensor=False)
        self.dequant_freq = qnn.QuantIdentity(return_quant_tensor=False)
        
        # --- Re-quantization Stub for Fusion ---
        # This will quantize the concatenated FP32 tensor to INT8
        # before feeding it to the INT8 fusion_mlp.
        self.quant_fusion_inp = qnn.QuantIdentity(
            act_quant=Int8Quant.Act, return_quant_tensor=True
        )

    def forward(self, x_spatial, x_freq):
        # x_spatial and x_freq are FP32
        
        # --- Spatial Stream (INT8) ---
        spatial_features_q = self.spatial_stream(self.quant_spatial_inp(x_spatial))
        
        # --- Frequency Stream (INT16) ---
        freq_features_map_q = self.frequency_stream(x_freq)

        # --- Fusion Logic (DEQUANT-CAT-QUANT) ---
        
        # 1. Dequantize both streams to FP32
        spatial_features = self.dequant_spatial(spatial_features_q)
        freq_features_map = self.dequant_freq(freq_features_map_q)
        
        # Reshape frequency features in FP32
        freq_features = freq_features_map.view(freq_features_map.size(0), -1)
        
        # 2. Concatenate in FP32
        fused_features = torch.cat((spatial_features, freq_features), dim=1)
        
        # 3. Re-quantize to INT8 for the MLP
        fused_features_q = self.quant_fusion_inp(fused_features)
        
        # --- Heads ---
        
        # Run INT8 Fusion MLP
        classification_output = self.fusion_mlp(fused_features_q)
        
        # Run INT16 Reconstruction Head
        # We use the INT16 output from the frequency stream directly
        reconstruction_output = self.reconstruction_head(freq_features_map_q)
        
        # Both outputs are FP32 (as specified by return_quant_tensor=False)
        return classification_output, reconstruction_output
