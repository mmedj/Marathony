# photograph.py
from flask import Blueprint, jsonify, request
import os
from ocr.predictWithOCR import predict
photograph_bp = Blueprint('photograph', __name__)

@photograph_bp.route('/photograph_dashboard', methods=['POST'])
def photograph_dashboard():
    # Photograph dashboard logic goes here
    return jsonify({'message': 'Photograph dashboard'})
UPLOAD_FOLDER = 'static/images/Untreated'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@photograph_bp.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files")

    filenames = []
    for file in uploaded_files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = file.filename
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filenames.append(filename)
    predict()
    return jsonify({'message': 'Files uploaded successfully', 'filenames': filenames})