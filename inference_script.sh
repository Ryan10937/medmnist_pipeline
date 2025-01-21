#!/bin/bash
docker build -f Dockerfile -t medmnist_inference .
docker run -v ./results:/app/results -e MODEL_PATH=models/model.keras -e DATA_FOLDER=inference_images -e IMAGE_SIZE=64 medmnist_inference