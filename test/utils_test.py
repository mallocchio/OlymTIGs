import torch
import numpy as np

def reshape(image):
    # Converti l'array numpy in un tensore PyTorch
    tensor = torch.tensor(image, dtype=torch.float32)
    # Flatten l'immagine in modo che abbia le dimensioni (784,)
    tensor = tensor.view(-1, 1, 28, 28)
    return tensor

def predict(model, image):
    image = reshape(image)
    with torch.no_grad():
        logps = model(image)
    ps = torch.exp(logps)
    prediction = clean_prediction(ps, logps)
    return prediction

def clean_prediction(ps, logps):
        probab = list(ps.numpy()[0])
        predicted_digit = probab.index(max(probab))
        ps = ps.data.numpy().squeeze()
        logps = logps.data.numpy().squeeze()
        return ps, logps, predicted_digit

def load_model(self, model_path):
    self.model = torch.load(model_path)
    return self.model

def tranform_image(image):
    img = apply_filters(image)
    img = add_noise(img)
    return img

def apply_filters(image):
    filtered_image = cv2.GaussianBlur(image, (5, 5), 0)
    filtered_image = cv2.Canny(image, 100, 200)
    return filtered_image

def add_noise(image, noise_factor=0.5):
    noise = np.random.randn(*image.shape) * noise_factor
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image