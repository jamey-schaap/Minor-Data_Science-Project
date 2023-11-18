FROM tensorflow/tensorflow:latest-jupyter
LABEL authors="jamey"

WORKDIR /app

RUN mkdir -p ./out

COPY ./datasets/MachineLearning-Dataset-V1.xlsx ./
COPY ./neural-network.ipynb ./
COPY ./requirements.txt ./

RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED 1
ENV pwd 1hoQyxpr5x9Wpy2MIJlN
EXPOSE 8888

ENTRYPOINT jupyter notebook --ip=0.0.0.0 --allow-root --NotebookApp.token=$pwd --NotebookApp.password=$pwd
