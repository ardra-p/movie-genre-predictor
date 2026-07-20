import numpy as np
from PIL import Image
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
import os

MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ['Action', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Horror', 'Romance', 'Thriller']

class PredictView(APIView):
    def post(self, request):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        img_file = request.FILES['image']
        img = Image.open(img_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)  # no /255.0 — EfficientNetB0 handles it
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)[0]

        result = []
        for i in range(len(predictions)):
            result.append({
                'genre': CLASS_NAMES[i],
                'confidence': round(float(predictions[i]) * 100, 1)
            })
       
        result = sorted(result, key=lambda x: x['confidence'], reverse=True)
        predicted = [
            {'genre': item['genre'], 'confidence': item['confidence']}
            for item in result if item['confidence'] > 50
        ]

        if not predicted:
            predicted = [{'genre': result[0]['genre'], 'confidence': result[0]['confidence']}]

        return Response({'predictions': predicted})