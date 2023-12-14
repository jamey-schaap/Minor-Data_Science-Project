# Data Science project

## Requirements

Python `3.11.x`<br/>

## Setting up

## Auto setup
Currently, this script only works if your `Python` bin is installed under `C:\Users\<user>\AppData\Local\Programs\`. To set the environment up, start a `Powershell 7 Core` `(PWS)` shell as administrator, navigate to the root directory of the project and run the script (`.\simple-setup.ps1`). 

### Manual steps
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
Update Pip <br/>
`python -m pip install --upgrade pip`
<br/>
<br/>
Install requirements <br/>
`pip install -r requirements.txt`


## During development

Dependent on which editor/IDE you use you might have to activate the virtual Python environment manually. This can be done with: <br/>
Unix: `source env/bin/activate` <br/>
PowerShell (PWSH) (Core): `venv\Scripts\Activate.ps1`<br/>
Command Prompt (CMD): `venv\Scripts\activate.bat`


