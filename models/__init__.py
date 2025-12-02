"""
Model package for COVID-19 Classification
"""
from models.vgg16_model import get_vgg16_model, VGG16Model
from models.resnet50_model import get_resnet50_model, ResNet50Model
from models.densenet121_model import get_densenet121_model, DenseNet121Model

__all__ = [
    'get_vgg16_model',
    'get_resnet50_model',
    'get_densenet121_model',
    'VGG16Model',
    'ResNet50Model',
    'DenseNet121Model'
]


def get_model(model_name, num_classes=3, pretrained=True):
    """
    Get model by name
    
    Args:
        model_name (str): Name of the model ('vgg16', 'resnet50', 'densenet121')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Model instance
    """
    model_name = model_name.lower()
    
    if model_name == 'vgg16':
        return get_vgg16_model(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet50':
        return get_resnet50_model(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'densenet121':
        return get_densenet121_model(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f'Unknown model: {model_name}. Choose from: vgg16, resnet50, densenet121')
