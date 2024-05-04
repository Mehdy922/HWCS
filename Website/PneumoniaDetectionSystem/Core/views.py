from django.http import HttpResponse
from django.shortcuts import render
from .models import UploadImage
from .utils import load_trained_model,predict_image
from joblib import load

def homePage(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            uploaded_image = request.FILES['image']
            image_path = 'uploads/' + uploaded_image.name
            with open(image_path, 'wb') as f:
                for chunk in uploaded_image.chunks():
                    f.write(chunk)

            model_path = 'E:\HWCS\knn_model.joblib'
            scaler_path = 'E:\HWCS\scaler.joblib'
            model = load_trained_model(model_path)
            scaler = load(scaler_path)

            predicted_label = predict_image(model, image_path, scaler)

            prediction_result = predicted_label

            return render(request, 'result.html', {'prediction_result': prediction_result})
        else:
            return HttpResponse('No image uploaded!')
    return render(request, 'home.html')

def result(request):
    return render(request, 'result.html')


