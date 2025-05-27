import os
import sys
from io import BytesIO

from django.conf import settings
from django.core.management import execute_from_command_line
from django.http import JsonResponse
from django.urls import path

import numpy as np
import face_recognition
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='insecure-secret-key',
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=['*'],
        MIDDLEWARE=[
            'django.middleware.common.CommonMiddleware',
        ],
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
        ],
    )

import django
django.setup()

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


def resize_image(image_np, max_width=250):
    height, width = image_np.shape[:2]
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        image_np = np.array(Image.fromarray(image_np).resize((max_width, new_height), Image.LANCZOS))
    return image_np


def load_image_from_url(url, max_width=250):
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    # Resize early here
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.LANCZOS)
    return np.array(image)


def load_image_from_file(file, max_width=250):
    image = Image.open(file)
    if image.width > max_width:
        ratio = max_width / image.width
        new_height = int(image.height * ratio)
        image = image.resize((max_width, new_height), Image.LANCZOS)
    return np.array(image)


@csrf_exempt
@require_http_methods(["POST"])
def compare_face(request):
    try:
        if not request.content_type.startswith('multipart/form-data'):
            return JsonResponse({"success": False, "message": "Use multipart/form-data with images."})

        # Get image1 either from URL or file
        image1_url = request.POST.get('image1_url')
        image1_file = request.FILES.get('image1')

        # Get image2 either from URL or file
        image2_url = request.POST.get('image2_url')
        image2_file = request.FILES.get('image2')

        if image1_url:
            try:
                image1_np = load_image_from_url(image1_url)
            except Exception as e:
                return JsonResponse({"success": False, "message": "Failed to load image1 from URL.", "error": str(e)})
        elif image1_file:
            try:
                image1_np = load_image_from_file(image1_file)
            except Exception as e:
                return JsonResponse({"success": False, "message": "Failed to load image1 from uploaded file.", "error": str(e)})
        else:
            return JsonResponse({"success": False, "message": "Provide image1_url or upload image1 file."})

        if image2_url:
            try:
                image2_np = load_image_from_url(image2_url)
            except Exception as e:
                return JsonResponse({"success": False, "message": "Failed to load image2 from URL.", "error": str(e)})
        elif image2_file:
            try:
                image2_np = load_image_from_file(image2_file)
            except Exception as e:
                return JsonResponse({"success": False, "message": "Failed to load image2 from uploaded file.", "error": str(e)})
        else:
            return JsonResponse({"success": False, "message": "Provide image2_url or upload image2 file."})

        # Use 'hog' for CPU or 'cnn' if GPU is available (change here if needed)
        model_name = 'hog'

        # Find face locations
        face_locations_1 = face_recognition.face_locations(image1_np, model=model_name)
        face_locations_2 = face_recognition.face_locations(image2_np, model=model_name)

        if not face_locations_1:
            return JsonResponse({"success": False, "message": "No face found in image1."})
        if not face_locations_2:
            return JsonResponse({"success": False, "message": "No face found in image2."})

        # Get encodings with num_jitters=1 for speed
        encoding1 = face_recognition.face_encodings(image1_np, known_face_locations=face_locations_1, num_jitters=1)[0]
        encoding2 = face_recognition.face_encodings(image2_np, known_face_locations=face_locations_2, num_jitters=1)[0]

        distance = np.linalg.norm(encoding1 - encoding2)
        threshold = 0.6
        match = bool(distance <= threshold)

        return JsonResponse({
            "success": True,
            "match": match,
            "distance": round(float(distance), 4),
            "threshold": threshold,
            "message": "Faces match" if match else "Faces do not match"
        })

    except Exception as e:
        return JsonResponse({
            "success": False,
            "message": "error_processing_request",
            "error": str(e),
            "error_type": str(type(e))
        })


urlpatterns = [
    path('compare-face/', compare_face),
]


if __name__ == "__main__":
    execute_from_command_line(sys.argv)


from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()


