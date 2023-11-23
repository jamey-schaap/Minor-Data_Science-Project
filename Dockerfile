FROM tensorflow/tensorflow:latest-jupyter
LABEL authors="jamey"

WORKDIR /app

RUN mkdir -p ./out

COPY ./datasets/MachineLearning-Dataset-V1.xlsx ./
COPY neural-network.ipynb ./
COPY ./requirements.txt ./

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PWD 1hoQyxpr5x9Wpy2MIJlN
ENV OUTPUT_PATH /app/out

EXPOSE 8888

ENTRYPOINT jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=$JUPYTER_PWD --NotebookApp.password=$JUPYTER_PWD --no-browser
