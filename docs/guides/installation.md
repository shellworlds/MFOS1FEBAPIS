# Installation Guide

## System Requirements

### Linux (Ubuntu 22.04+)
- 32GB RAM minimum, 64GB recommended
- NVIDIA GPU with CUDA 12.0+ (for HPC features)
- 100GB free disk space

### macOS (13+)
- Apple Silicon or Intel
- 32GB RAM recommended
- 50GB free disk space

### Windows
- WSL2 with Ubuntu 22.04
- 32GB RAM recommended
- 100GB free disk space

## Quick Install

### Linux/macOS
```bash
curl -sSL https://raw.githubusercontent.com/shellworlds/MFOS1FEBAPIS/main/client_installers/universal/install.sh | bash
Windows (WSL2)
powershell
wsl --install -d Ubuntu
curl -sSL https://raw.githubusercontent.com/shellworlds/MFOS1FEBAPIS/main/client_installers/windows/install_wsl.ps1 | powershell -c -
Verification
python
from mfos1febapis import test_installation
test_installation()
