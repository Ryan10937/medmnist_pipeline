# Getting Started
Clone this repo and cd to the medmnist_pipeline folder
Navigate to a new python environment and pip install the requirements file in install folder using:
pip install -r install/requirement.txt


## Training

Run "train_model.sh" and wait for model training to complete

## Inference

run "build_inference_app.sh"

Go to your browser and type "http://localhost:9000"

Upload an image using the buttons shown

then add "/infer" to the end of the URI

The result will be a dataframe with predictions next to the image name
