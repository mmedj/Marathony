# user.py
from flask import Blueprint, request, jsonify
import os

user_bp = Blueprint('user', __name__)


def find_folder_by_number(number, base_dir):
    for folder_name in os.listdir(base_dir):
        if folder_name.isdigit() and int(folder_name) == number:
            folder_path = os.path.join(base_dir, folder_name)
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if
                      img.endswith(('jpg', 'jpeg', 'png', 'gif'))]
            return images
    return None


@user_bp.route('/user_dashboard', methods=['GET'])
def photograph_dashboard():
    # Photograph dashboard logic goes here
    return jsonify({'message': 'User dashboard'})
@user_bp.route('/find_folder', methods=['GET'])
def find_folder():
    number = request.args.get('number')
    base_dir = 'static/images/Treated'  # Replace this with your actual base directory

    if number is None:
        return jsonify({'error': 'Number parameter is missing.'}), 400

    try:
        number = int(number)
    except ValueError:
        return jsonify({'error': 'Number parameter must be an integer.'}), 400

    images = find_folder_by_number(number, base_dir)
    if images:
        return jsonify({'images': images}), 200
    else:
        return jsonify({'error': 'Folder not found or no images found in the folder.'}), 404
