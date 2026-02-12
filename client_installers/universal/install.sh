#!/bin/bash
set -e

echo "MFOS1FEBAPIS Quantum Framework Installation"
echo "System: $(uname -a)"
echo "================================"
if [ ! -f /etc/os-release ]; then
echo "Unsupported OS. This installer requires Ubuntu/Debian."
exit 1
fi


sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y build-essential cmake git wget curl
python3 python3-pip python3-venv
libopenblas-dev liblapack-dev
libssl-dev libffi-dev
llvm-14-dev
openjdk-17-jdk
golang-go
nodejs npm
clangd-14 lldb-14


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update -y
sudo apt-get install -y cuda-quantum-0.8
python3 -m venv mfos_env
source mfos_env/bin/activate


pip install --upgrade pip setuptools wheel
pip install qiskit==1.0.2 qiskit-ibm-runtime==0.22.0 qiskit-nature==0.7.2
pip install mitiq==0.30.0 pennylane==0.36.0 pennylane-qiskit==0.36.0
pip install amazon-braket-sdk amazon-braket-pennylane-plugin
pip install azure-quantum==0.29.0
pip install openfermion==1.6.0 pyscf==2.6.0
pip install tensorflow==2.16.1 tensorflow-quantum==0.7.2
pip install torch==2.2.0 torchvision==0.17.0
pip install jax jaxlib
pip install numpy==1.26.3 scipy==1.12.0 pandas==2.2.0
pip install matplotlib==3.8.2 seaborn==0.13.2 plotly==5.18.0
pip install scikit-learn==1.4.0 scikit-image==0.22.0
pip install pymatgen==2024.1.1 ase==3.22.1
pip install fastapi==0.109.0 uvicorn==0.27.0
npm install -g next vite react react-dom
cd ~
git clone https://github.com/shellworlds/MFOS1FEBAPIS.git
cd MFOS1FEBAPIS
pip install -e .

echo "Installation complete. Run 'python -c "from mfos1febapis import test_installation; test_installation()"' to verify."
