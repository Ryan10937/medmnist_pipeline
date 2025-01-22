#!/bin/bash
docker build -f Dockerfile -t medmnist_inference .
docker run -v ./results:/app/results -p 9000:9000 medmnist_inference 