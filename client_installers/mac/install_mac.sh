#!/bin/bash
set -e

echo "MFOS1FEBAPIS Quantum Framework Installation for macOS"
echo "System: $(uname -m)"
echo "================================"
if ! command -v brew &> /dev/null; then
echo "Installing Homebrew..."
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
brew install python@3.11 cmake llvm libomp openblas lapack
brew install node npm go openjdk@17
python3 -m venv mfos_env
source mfos_env/bin/activate
pip install --upgrade pip
pip install qiskit pennylane amazon-braket-sdk azure-quantum
echo "macOS installation complete."
