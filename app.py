from flask import Flask, render_template, request, redirect, url_for
from pdf2image import convert_from_path
import os

from paddleocr import PaddleOCR

from PIL import Image, ImageDraw

import numpy as np
import pandas as pd

import time

import math
from typing import Tuple, Union
from statistics import mode, mean

import re
from fuzzywuzzy import fuzz

from utils import TranscriptOCR

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

transcriptocr = TranscriptOCR()

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        # Save the uploaded file as test.jpg
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            filename = 'test.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        elif file.filename.lower().endswith('.pdf'):
            # Convert the first page of the PDF to an image and save it as test.jpg
            filename = 'test.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pages = convert_from_path(file, 500)
            pages[0].convert('RGB').save(file_path, 'JPEG')

        # Load the saved image and process it using draw_image
        processed_image = transcriptocr.draw_boxes(file_path)
        processed_image.convert('RGB').save(os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg'))

        json_data = transcriptocr.extract_json(file_path)

        return render_template('result.html', processed_image=processed_image, json_data=json_data)
    else:
        return 'Invalid file format'

if __name__ == '__main__':
    app.run(debug=True)