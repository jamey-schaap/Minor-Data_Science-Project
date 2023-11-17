FROM tensorflow/tensorflow:latest
LABEL authors="jamey"

WORKDIR /app

RUN mkdir -p ./out

COPY ./datasets/MachineLearning-Dataset-V1.xlsx ./
COPY ./neural-network.py ./
COPY ./requirements.txt ./

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1
EXPOSE 8080

ENTRYPOINT ["python", "neural-network.py"]
