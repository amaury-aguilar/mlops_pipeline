#!/bin/bash

echo ""
echo "=== Python Virtual Environment Setup ==="
echo ""

ENV_NAME="mlops-venv"

echo "Creating virtual environment..."

python3 -m venv $ENV_NAME

echo "Activating virtual environment..."

source $ENV_NAME/bin/activate

echo ""
echo "Installing requirements..."

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Registering Jupyter kernel..."

python -m ipykernel install --user 
--name=$ENV_NAME 
--display-name="$ENV_NAME"

echo ""
echo "Setup completed successfully!"
