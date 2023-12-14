[string] $python_path = "C:\Users\${Env:Username}\AppData\Local\Programs\Python\Python311\python.exe"

Invoke-Expression "pip install virtualenv"
Invoke-Expression "${python_path} -m venv .\venv"
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt