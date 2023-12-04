FROM jupyter/tensorflow-notebook:x86_64-python-3.11.6
LABEL authors="jamey"

WORKDIR /app

RUN mkdir -p ./out

COPY ./ ./

RUN pip install -r requirements.txt
RUN pip install shap==0.43.0

ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PWD 1hoQyxpr5x9Wpy2MIJlN
ENV OUTPUT_PATH /app/out

EXPOSE 8888

ENTRYPOINT jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.token=$JUPYTER_PWD --NotebookApp.password=$JUPYTER_PWD --no-browser
