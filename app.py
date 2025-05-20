import gradio as gr
import torch
import pickle
from PIL import Image
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import os

# Model yükleme
def load_model():
    with open('scene_classifier.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = EfficientNet.from_pretrained('efficientnet-b0')

    # Eğitimdeki yapıya tam uyacak şekilde tanımla
    model._fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model._fc.in_features, 6)
    )

    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    return model, model_data['class_to_idx']

# Tahmin fonksiyonu
def predict(image):
    model, class_to_idx = load_model()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predictions = {idx_to_class[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return predictions

# Örnek görüntüleri yükle
def load_example_images():
    examples = []
    dataset_path = "dataset/seg_train"
    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    for category in categories:
        img_path = os.path.join(dataset_path, category, "0.jpg")
        if os.path.exists(img_path):
            examples.append([img_path])
    
    return examples

# Gradio arayüzü
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=6),
    title="Scene Classification",
    description="Upload an image to classify it into one of these categories: buildings, forest, glacier, mountain, sea, or street.",
    examples=load_example_images()  # Örnek görüntüleri ekle
)

if __name__ == "__main__":
    iface.launch()