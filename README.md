# Data Science project

## Requirements

Python `>= 3.12.0`<br/>

## Setting up

Install virtualenv for Python <br/>
`pip install virtualenv`
<br/>
<br/>
Create a virtual python environment <br/>
`python -m venv .\venv`
<br/>
<br/>
Activate the virtual environment <br/>
Unix: `source env/bin/activate` <br/>
PowerShell (PWSH) (Core): `venv\Scripts\Activate.ps1`<br/>
Command Prompt (CMD): `venv\Scripts\activate.bat`
<br/>
<br/>
Install requirements <br/>
`pip install -r requirements.txt`
<br/>
<br/>
Update Pip <br/>
`python -m pip install --upgrade pip`

## During development

Dependent on which editor/IDE you use you might have to activate the virtual Python environment manually. This can be done with: <br/>
Unix: `source env/bin/activate` <br/>
PowerShell (PWSH) (Core): `venv\Scripts\Activate.ps1`<br/>
Command Prompt (CMD): `venv\Scripts\activate.bat`

## Tensorflow with Docker and Jupyter
Run `docker compose up [-d]` to create and start the Docker container. The Jupyter server can be reached on `https://127.0.0.1:8888`, enter `1hoQyxpr5x9Wpy2MIJlN` in the password/token field to gain access. Pycharm can connect to the Jupyter server with the following URL: `http://127.0.0.1:8888/?token=1hoQyxpr5x9Wpy2MIJlN`.

