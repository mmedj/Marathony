# app.py
from flask import Flask
from user import user_bp
from photograph import photograph_bp
from flask_cors import CORS

app = Flask(__name__, static_url_path='/static')
CORS(app)  # This will enable CORS for all routes in your Flask app


# Register blueprints
app.register_blueprint(user_bp, url_prefix='/user')
app.register_blueprint(photograph_bp, url_prefix='/photograph')

if __name__ == '__main__':
    app.run(debug=True)
