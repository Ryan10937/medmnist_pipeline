# FROM tensorflow/tensorflow:2.18.0 AS build
# WORKDIR /app
# COPY requirements.txt /app/
# RUN sudo pip install --no-cache-dir -r requirements.txt

# EXPOSE 5000
# ENV EPOCHS=$EPOCHS
# ENV IMAGE_SIZE=$IMAGE_SIZE
# ENV BATCH_SIZE=$BATCH_SIZE
# ENV DATASET_NAME=$DATASET_NAME
# ENTRYPOINT ["python", "/app/train.py","--epochs", "$EPOCHS","--image_size", "$IMAGE_SIZE","--batch_size", "$BATCH_SIZE","--dataset_name", "$DATASET_NAME"]

FROM ryan10937/train_image1
WORKDIR /app
COPY . /app
VOLUME /app/results

EXPOSE 5000

ENTRYPOINT python /app/infer.py --model_path $MODEL_PATH --data_folder $DATA_FOLDER --image_size $IMAGE_SIZE
