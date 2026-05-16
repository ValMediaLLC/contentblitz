#!/usr/bin/env bash

# Developer environment bootstrap script

set -e

echo
echo "=========================================="
echo "  ContentBlitz AMD64 Python Setup"
echo "=========================================="
echo

echo "Creating .venv-x64..."
py -3.13 -m venv .venv-x64

if [ ! -f ".venv-x64/Scripts/python.exe" ]; then
    echo "Failed to create .venv-x64"
    exit 1
fi

echo
echo "Verifying architecture..."
./.venv-x64/Scripts/python.exe -c "import sysconfig; print(sysconfig.get_platform())"

echo
echo "Upgrading pip..."
./.venv-x64/Scripts/python.exe -m pip install --upgrade pip

echo
echo "Installing requirements..."
./.venv-x64/Scripts/python.exe -m pip install -r requirements.txt

echo
echo "Activating virtual environment..."
echo

source .venv-x64/Scripts/activate

echo
echo "Environment ready."
echo