import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import json
import cv2
from ocr.externals.detect_text import TextDetection
from ocr.externals.tesseract import Tesseract
from configs import config


text_detection = TextDetection(weights=config["text_detection_model"])
text_detection.load_model()
obj = Tesseract()


def index(request):
    response = json.dumps([{}])
    return HttpResponse(response, content_type='text/json')

@csrf_exempt
def inference(request):
    response = {}
    if request.method == 'POST':

        image_url = request.GET.get('image')
        
        boxes = text_detection.detect(source=image_url, img_size=config["img_size"])
        try:
            response = obj.inference(image=cv2.imread(image_url), boxes=boxes)
        except:
            response = json.dumps([{ 'Error': 'Car could not be added!'}])
    return HttpResponse(response, content_type='text/json')
