__author__ = "JasonLuo"
from torchcam.methods import LayerCAM
from torchvision.models import resnet18
from PIL import Image
import torchvision.utils as vutils
import torchvision.transforms.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.densenet169(pretrained=True)


num_features = model.classifier.in_features
model.classifier = torch.nn.Sequential(
                torch.nn.Linear(num_features, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 2),
)

model.to(device)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = r'C:\Users\Woody\Desktop\Lay_visualization\Normal\225973.png.png'
image = Image.open(img_path).convert('RGB')  # Replace with your image path
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = preprocess(image).unsqueeze(0)

cam_extractor = LayerCAM(model)

out = model(img_tensor)

activation_map = cam_extractor(out[0].argmax().item(),out)
