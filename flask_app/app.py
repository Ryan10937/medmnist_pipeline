import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import subprocess
import pandas as pd
app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'file' not in request.files:
        return 'No file part', 400

    files = request.files.getlist('file')

    # If no files are selected
    if len(files) == 0 or any(file.filename == '' for file in files):
        return 'No selected file', 400

    # Save the images
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

    return render_template('home.html', message="Images uploaded successfully!")

@app.route('/infer')
def infer():
    result = subprocess.run(['./infer_on_data_folder.sh'], check=True, capture_output=True, text=True)
    results_df = pd.read_csv('results/predictions.csv')
    df_html = results_df.to_html(classes='table table-bordered', index=False)
    
    # Render the DataFrame as an HTML page
    return render_template('display_dataframe_results.html', df_html=df_html)
@app.route('/stop', methods=['GET'])
def stop():
    os._exit(0)
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=9000)
