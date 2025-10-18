# CUDA Setup Notes - Updated October 18, 2025

## System Information
- **GPU**: NVIDIA GeForce RTX 2060 SUPER
- **NVIDIA Driver Version**: 581.57 (Updated)
- **Supported CUDA Version**: 13.0
- **Operating System**: Windows

## Current Installation ✅
**PyTorch version:** 2.6.0+cu124  
**CUDA available:** True  
**CUDA version:** 12.4  
**cuDNN version:** 90100

Successfully installed PyTorch with CUDA 12.4 support in the virtual environment.

## Installation History

### Initial Setup (Driver 511.79)
Initially had NVIDIA driver 511.79 which only supported CUDA 11.6. Installed PyTorch with CUDA 11.8 for compatibility.

### Updated Setup (Driver 581.57)
After updating NVIDIA drivers to version 581.57 (supporting CUDA 13.0), upgraded to PyTorch with CUDA 12.4 support.

## Installation Command
```powershell
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Virtual Environment
- **Location**: `D:\algo-drl-sac-iql\.venv`
- **Python Version**: 3.12.10
- **Python Executable**: `D:\algo-drl-sac-iql\.venv\Scripts\python.exe`

## Verification Command
To verify CUDA is working in your venv:

```powershell
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -c "import torch; print('is_available:', torch.cuda.is_available()); [print('device:', torch.cuda.get_device_name(0), 'CUDA', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()) for _ in [None] if torch.cuda.is_available()]"
```

**Current Output:**
```
is_available: True
device: NVIDIA GeForce RTX 2060 SUPER CUDA 12.4 cudnn 90100
```

## Important Notes

### CUDA Toolkit Not Required
PyTorch bundles its own CUDA runtime libraries, so you **don't need to install the CUDA Toolkit** or add CUDA to your PATH for PyTorch to work. The PyTorch wheel includes everything needed.

### Driver Compatibility
- Current driver (581.57) supports CUDA 13.0
- PyTorch CUDA 12.4 is fully compatible with this driver
- CUDA 12.4 is the latest PyTorch version available (as of October 2025)

### Available PyTorch CUDA Versions
- **CUDA 12.4** (Latest) - Requires driver 525.60 or newer ✅ Currently installed
- **CUDA 12.1** - Requires driver 525.60 or newer
- **CUDA 11.8** - Requires driver 450.80 or newer

To install a different version, use:
```powershell
# CUDA 12.1
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage in Your Trading Algo
When running your scripts, make sure to use the virtual environment's Python:

```powershell
# Method 1: Use full path to venv Python
D:\algo-drl-sac-iql\.venv\Scripts\python.exe src\run_offline_pretrain.py

# Method 2: Activate venv first (if execution policy allows)
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# D:\algo-drl-sac-iql\.venv\Scripts\Activate.ps1
# python src\run_offline_pretrain.py
```

## Quick GPU Test
To test that GPU tensor operations work:

```powershell
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -c "import torch; x = torch.rand(3, 3).cuda(); y = torch.rand(3, 3).cuda(); z = x @ y; print('GPU tensor operation successful!'); print('Result device:', z.device)"
```

**Expected output:**
```
GPU tensor operation successful!
Result device: cuda:0
```

## Troubleshooting

### If CUDA is not available:
1. Check driver version: `nvidia-smi`
2. Verify PyTorch version: `D:\algo-drl-sac-iql\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"`
3. Should show `2.6.0+cu124` (the `+cu124` indicates CUDA support)
4. If it shows just `2.x.0` without `+cu`, you have the CPU-only version

### Reinstall if needed:
```powershell
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
D:\algo-drl-sac-iql\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Performance Benefits
With CUDA 12.4 enabled on your RTX 2060 SUPER, you can expect:
- **10-100x faster** neural network training compared to CPU
- Support for larger batch sizes (8GB VRAM available)
- Efficient deep reinforcement learning training for your trading algorithms
- Full support for d3rlpy offline RL algorithms with GPU acceleration
