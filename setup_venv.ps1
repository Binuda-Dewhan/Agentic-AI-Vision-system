# Create a virtual environment
Write-Host "Creating virtual environment in .\venv..."
python -m venv venv

# Activate the virtual environment
Write-Host "Activating virtual environment..."
$env:VIRTUAL_ENV="$PWD\venv"
$env:Path="$PWD\venv\Scripts;$env:Path"

# Install requirements
Write-Host "Installing requirements (this may take a few minutes)..."
python -m pip install --upgrade pip
python -m pip install -e .

Write-Host "Setup complete! To activate the environment in the future, run: .\venv\Scripts\Activate.ps1"
