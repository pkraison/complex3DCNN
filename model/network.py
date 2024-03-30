import torch
import torch.nn as nn
'''Complex value 3D CNN'''

class ComplexConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv3d, self).__init__()
        self.real_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, complex_input):
        # complex_input is expected to be a tuple (real_part, imag_part)
        real, imag = complex_input
        # Apply the convolutions separately and combine
        real_part = self.real_conv(real) - self.imag_conv(imag)
        imag_part = self.real_conv(imag) + self.imag_conv(real)
        return real_part, imag_part
class ComplexConv3dNet(nn.Module):
    def __init__(self):
        super(ComplexConv3dNet, self).__init__()
        self.layer1 = ComplexConv3d(192, 96, kernel_size=3, stride=1, padding=1)
        self.layer2 = ComplexConv3d(96, 48, kernel_size=3, stride=1, padding=1)
        self.layer3 = ComplexConv3d(48, 24, kernel_size=3, stride=1, padding=1)
        self.layer4 = ComplexConv3d(24, 12, kernel_size=3, stride=1, padding=1)
        # Output layer to map to real values; adjust as necessary for your task
        # This example uses a simple flattening and linear layer as a placeholder
        self.flatten = lambda x: torch.flatten(x, 1)  # Simplified flattening
        self.output_layer = nn.Linear(whatever_the_flattened_size_is, output_size)
    
    def forward(self, x):
        # x is a tuple (real_part, imag_part)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Assuming some form of non-linearity and pooling might be applied between layers
        # Flatten and pass through the output layer
        # Here, we handle only the real part as an example
        real, imag = x
        real_flat = self.flatten(real)
        output = self.output_layer(real_flat)  # Example: using real part for output
        return output
