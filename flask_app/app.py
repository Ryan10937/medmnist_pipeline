from flask import Flask, request, redirect, url_for, render_template
import os
import subprocess

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for uploading images
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Route for handling the file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # Call inference.py to process the uploaded image
        result = subprocess.run(['python', 'inference.py', filename], capture_output=True, text=True)
        
        # Handle the output from the inference script (result.stdout)
        return f'Inference result: {result.stdout}'
    else:
        return 'Invalid file type. Only images are allowed.'

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
