from flask import Flask, request, redirect, url_for, render_template,render_template_string
import os
import subprocess
import pandas as pd
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
        return f'Image uploaded'
        # return f'Inference result: {result.stdout}'
    else:
        return 'Invalid file type. Only images are allowed.'
@app.route('/infer', methods=['GET'])
def infer():
    result = subprocess.run(['./inference_script.sh'], check=True, capture_output=True, text=True)
    results_df = pd.read_csv('results/predictions.csv')
    df_html = results_df.to_html(classes='table table-bordered', index=False)
    
    # Render the DataFrame as an HTML page
    return render_template_string("""
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>DataFrame Display</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container mt-5">
                <h1>Displaying Pandas DataFrame</h1>
                <div class="table-responsive">
                    {{ df_html|safe }}
                </div>
            </div>
        </body>
        </html>
    """, df_html=df_html)

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True,host='0.0.0.0',port=9000)

