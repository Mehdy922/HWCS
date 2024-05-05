from joblib import load
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image as Img, ImageOps

def load_trained_model(model_path):
    loaded_model = load(model_path)
    return loaded_model

def extract_glcm_features(image_path):
    with Image.open(image_path) as image:
        grayscale_image = image.convert('L')
        grayscale_image_np = np.array(grayscale_image)
        glcm = graycomatrix(grayscale_image_np, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        return [contrast, dissimilarity, homogeneity, energy, correlation]


def predict_image(model, img_path, scaler):
        features = extract_glcm_features(img_path)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        print(f"Predicted writer for the provided handwriting is {prediction}")

        return prediction

