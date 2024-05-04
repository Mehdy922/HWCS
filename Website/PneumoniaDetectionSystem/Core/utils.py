from joblib import load
from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image as Img, ImageOps

def load_trained_model(model_path):
    loaded_model = load(model_path)
    return loaded_model

def extract_glcm_features(image):
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
    with Image.open(img_path) as img:
        grayscale_image = img.convert('L')
        target_size = (550, 720)
        padded_image = ImageOps.pad(grayscale_image,target_size, method=0, centering=(0.5, 0.5))
        width, height = padded_image.size
        block_size = 192
        top_left_x = (width - block_size) // 2
        top_left_y = (height - block_size) // 2
        center_block = padded_image.crop((top_left_x, top_left_y, top_left_x + block_size, top_left_y + block_size))

        features = extract_glcm_features(center_block)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        print(f"Predicted writer for the provided handwriting is {prediction}")

        return prediction

