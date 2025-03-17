import torch
import torch.nn as nn
from ICA_Detection.external.ultralytics.ultralytics.nn.tasks import attempt_load_one_weight
from ICA_Detection.external.ultralytics.ultralytics.engine.model import Model

class InputChannelAdapter(nn.Module):
    def __init__(self, original_model, in_channels=4):
        super(InputChannelAdapter, self).__init__()
        
        # Store original model
        self.model = original_model
        
        # Get the first conv layer
        first_conv = None
        for module in self.model.model.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                first_conv = module
                break
        
        if first_conv is None:
            raise ValueError("Could not find a Conv2d layer with 3 input channels")
        
        # Create a new conv layer with in_channels inputs but same output dimension
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Initialize the new weights: copy original weights for the first 3 channels
        # and initialize the mask channel weights to the average of RGB weights
        with torch.no_grad():
            # Copy weights for RGB channels
            new_conv.weight[:, :3] = first_conv.weight
            
            # Initialize mask channel (additional channel) with the average of RGB weights
            if in_channels > 3:
                new_conv.weight[:, 3:] = first_conv.weight.mean(dim=1, keepdim=True)
            
            # Copy bias if it exists
            if first_conv.bias is not None:
                new_conv.bias = nn.Parameter(first_conv.bias.clone())
        
        # Replace the first conv layer
        self._replace_first_conv(new_conv)
    
    def _replace_first_conv(self, new_conv):
        """Replace the first convolutional layer in the model with a new one."""
        # Find and replace the first Conv2d layer with 3 input channels
        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                # Get the parent module
                parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                child_name = name.rsplit('.', 1)[1] if '.' in name else name
                
                if parent_name:
                    parent = self.model.model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, new_conv)
                else:
                    setattr(self.model.model, child_name, new_conv)
                break
    
    def forward(self, x):
        return self.model(x)