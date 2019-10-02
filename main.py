import torchf
from collections import OrderedDict


class modifiedDeepLab(nn.module):
    def __init__(self, pretrained=True):
        """
        IntermediateLayerGetter is the general base model, or backbone
        DeepLabHead includes ASPP
        FCNHead is final processing of output from ASPP and outputs logits
        """
        self.__base_model = torch.hub.load('pytorch/vision:v0.4.0', 'deeplabv3_resnet101', pretrained=pretrained)
        self.__children_modules = {child.__class__.__name__: child for child in self.__base_model.children()}
        
        

