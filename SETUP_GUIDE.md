# 🚀 LLM 課程環境設置完整指南

## 📋 目錄
- [系統需求](#系統需求)
- [安裝步驟](#安裝步驟)
  - [Windows 環境設置](#windows-環境設置)
  - [Mac 環境設置](#mac-環境設置) 
  - [Linux 環境設置](#linux-環境設置)
- [GPU 設置指南](#gpu-設置指南)
- [雲端環境替代方案](#雲端環境替代方案)
- [環境驗證](#環境驗證)
- [常見問題解決](#常見問題解決)

---

## 系統需求

### 最低配置
- **CPU**: Intel i5 或 AMD Ryzen 5 以上
- **記憶體**: 8GB RAM
- **硬碟空間**: 20GB 可用空間
- **作業系統**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+

### 建議配置
- **CPU**: Intel i7 或 AMD Ryzen 7 以上
- **記憶體**: 16GB RAM 以上
- **GPU**: NVIDIA GTX 1060 6GB 以上 (選配)
- **硬碟空間**: 50GB 可用空間

---

## 安裝步驟

### Windows 環境設置

#### 步驟 1: 安裝 Python
1. 前往 [Python 官網](https://www.python.org/downloads/)
2. 下載 Python 3.9 或更新版本（建議 3.9.x）
3. **重要**: 安裝時勾選 "Add Python to PATH"
4. 安裝完成後，開啟命令提示字元，輸入：
```bash
python --version
```
應該顯示 Python 3.9.x 或更高版本

#### 步驟 2: 安裝 Visual Studio Build Tools
某些 Python 套件需要 C++ 編譯器：
1. 下載 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 執行安裝程式
3. 勾選「使用 C++ 的桌面開發」
4. 安裝（約需 5-10 分鐘）

#### 步驟 3: 創建虛擬環境
```bash
# 開啟命令提示字元或 PowerShell
cd C:\你的工作目錄

# 創建虛擬環境
python -m venv llm_env

# 啟動虛擬環境
llm_env\Scripts\activate

# 看到 (llm_env) 前綴表示啟動成功
```

#### 步驟 4: 安裝必要套件
```bash
# 更新 pip
python -m pip install --upgrade pip

# 安裝基礎套件
pip install numpy pandas matplotlib jupyter notebook

# 安裝 PyTorch (CPU 版本)
pip install torch torchvision torchaudio

# 安裝 Transformers 和相關套件
pip install transformers sentencepiece protobuf accelerate

# 安裝其他工具
pip install python-dotenv requests beautifulsoup4
```

---

### Mac 環境設置

#### 步驟 1: 安裝 Homebrew（如果還沒安裝）
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 步驟 2: 安裝 Python
```bash
# 安裝 Python
brew install python@3.9

# 驗證安裝
python3 --version
```

#### 步驟 3: 創建虛擬環境
```bash
# 進入工作目錄
cd ~/你的工作目錄

# 創建虛擬環境
python3 -m venv llm_env

# 啟動虛擬環境
source llm_env/bin/activate
```

#### 步驟 4: 安裝套件
```bash
# 更新 pip
pip install --upgrade pip

# 安裝所有必要套件
pip install numpy pandas matplotlib jupyter notebook
pip install torch torchvision torchaudio
pip install transformers sentencepiece protobuf accelerate
pip install python-dotenv requests beautifulsoup4
```

---

### Linux 環境設置

#### Ubuntu/Debian 系統
```bash
# 更新套件列表
sudo apt update

# 安裝 Python 和相關工具
sudo apt install python3.9 python3.9-venv python3-pip

# 創建虛擬環境
python3.9 -m venv llm_env

# 啟動虛擬環境
source llm_env/bin/activate

# 安裝套件
pip install --upgrade pip
pip install numpy pandas matplotlib jupyter notebook
pip install torch torchvision torchaudio
pip install transformers sentencepiece protobuf accelerate
```

---

## GPU 設置指南

### NVIDIA GPU 設置（選配）

#### 檢查 GPU 是否可用
```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")
```

#### 安裝 CUDA 版本的 PyTorch
1. 檢查您的 NVIDIA 驅動程式版本
2. 前往 [PyTorch 官網](https://pytorch.org/get-started/locally/)
3. 選擇對應的配置，獲取安裝命令

範例（CUDA 11.8）：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 雲端環境替代方案

### 🌟 Google Colab（推薦新手）

無需本地安裝，免費使用 GPU！

1. 前往 [Google Colab](https://colab.research.google.com/)
2. 登入 Google 帳號
3. 建立新筆記本
4. 選擇執行階段 > 變更執行階段類型 > GPU
5. 執行以下程式碼安裝套件：

```python
!pip install transformers accelerate sentencepiece
!pip install openai python-dotenv
```

### Kaggle Notebooks
1. 前往 [Kaggle](https://www.kaggle.com/)
2. 註冊帳號
3. 建立新 Notebook
4. 設定 > Accelerator > GPU

### Paperspace Gradient
- 提供免費 GPU 時數
- 預裝機器學習環境

---

## 環境驗證

創建檔案 `test_environment.py`：

```python
"""
環境測試腳本
執行此腳本驗證環境是否設置成功
"""

import sys
import importlib

def check_package(package_name):
    """檢查套件是否安裝"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} 已安裝")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安裝")
        return False

def main():
    print("=" * 50)
    print("LLM 課程環境檢查")
    print("=" * 50)
    
    # Python 版本
    print(f"\nPython 版本: {sys.version}")
    
    # 檢查必要套件
    packages = [
        "numpy",
        "pandas", 
        "torch",
        "transformers",
        "sentencepiece",
        "jupyter"
    ]
    
    print("\n檢查套件安裝狀態：")
    all_installed = True
    for package in packages:
        if not check_package(package):
            all_installed = False
    
    # 檢查 GPU
    print("\nGPU 檢查：")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU 可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            print("ℹ️  GPU 不可用，將使用 CPU")
    except:
        print("⚠️  無法檢查 GPU 狀態")
    
    # 結果
    print("\n" + "=" * 50)
    if all_installed:
        print("✅ 環境設置成功！可以開始課程了！")
    else:
        print("⚠️  請安裝缺少的套件")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

執行測試：
```bash
python test_environment.py
```

---

## 常見問題解決

### 問題 1: pip install 速度很慢
**解決方案：使用國內鏡像源**
```bash
# 使用清華大學鏡像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers
```

### 問題 2: ImportError: No module named 'xxx'
**解決方案：**
1. 確認虛擬環境已啟動
2. 重新安裝套件：
```bash
pip uninstall 套件名稱
pip install 套件名稱
```

### 問題 3: CUDA out of memory
**解決方案：**
1. 減少批次大小 (batch_size)
2. 使用較小的模型
3. 清理 GPU 記憶體：
```python
import torch
torch.cuda.empty_cache()
```

### 問題 4: Windows 上的編碼錯誤
**解決方案：**
設置環境變數：
```bash
set PYTHONIOENCODING=utf-8
```

### 問題 5: Mac M1/M2 晶片相容性
**解決方案：**
安裝 Metal 優化版本：
```bash
pip install torch torchvision torchaudio
```

### 問題 6: 虛擬環境無法啟動
**Windows PowerShell 解決方案：**
```powershell
# 執行政策設置
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 問題 7: Jupyter Notebook 無法開啟
**解決方案：**
```bash
# 重新安裝
pip uninstall jupyter notebook
pip install jupyter notebook

# 手動指定 port
jupyter notebook --port=8889
```

---

## 📞 需要協助？

如果遇到無法解決的問題：

1. **記錄錯誤訊息**：完整複製錯誤訊息
2. **檢查環境**：執行 `test_environment.py`
3. **查看版本**：記錄 Python 和套件版本
4. **課堂求助**：將以上資訊提供給講師

---

## 下一步

環境設置完成後，您可以：
1. 執行 `examples/02_environment/check_gpu.py` 測試環境
2. 開啟 Jupyter Notebook 開始互動式學習
3. 依序執行範例程式碼

祝學習愉快！🎉