# Search for a python 3.11.x version
[string[]] $pythonVersions = py -0p
[string[]] $python311Versions = $pythonVersions | Where-Object { ($_.split() | Where-Object { $_ })[0] -match "3.11" }

if ($python311Versions.length -le 0) {
  Write-Error "No python 3.11 versions found, please install python with version >= 3.11.0"
}

# Select the highest python versions
[string[]] $pythonVersion = $python311Versions.split() | Where-Object { $_ }
Write-Output "Found python version $($pythonVersion[0].TrimStart("-V:")) at $($pythonVersion[-1])"

# Install the virtual environment
pip install virtualenv
Invoke-Expression "$($pythonVersion[-1]) -m venv .\venv"

# Activate and enter the venv shell
.\venv\Scripts\Activate.ps1

# Setup the virtual environment
python -m pip install --upgrade pip
pip install -r requirements.txt