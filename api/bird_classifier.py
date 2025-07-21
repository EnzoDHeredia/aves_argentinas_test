import torch
from torchvision.models import efficientnet_b1

class BirdClassifier:
    def __init__(self, device, num_classes, model_path):
        self.device = device
        self.model = self._load_model(num_classes, model_path)

    def _load_model(self, num_classes, model_path):
        model = efficientnet_b1(weights=None)
        # Primero iguala la capa final a la cantidad de clases del checkpoint (102)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 102)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        # Si el número de clases es diferente, reemplaza la capa final
        if num_classes != 102:
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, tensor):
        """Realiza la predicción usando el modelo"""
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted].item()
        return predicted.item(), confidence
