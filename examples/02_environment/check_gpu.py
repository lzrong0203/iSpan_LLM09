"""
GPU 環境檢查工具
檢查系統是否有可用的 GPU 以及相關資訊
"""

import torch


def check_gpu_environment():
    """檢查並顯示 GPU 環境資訊"""
    
    print("=" * 50)
    print("GPU 環境檢查")
    print("=" * 50)
    
    # PyTorch 版本
    print(f"PyTorch版本: {torch.__version__}")
    
    # CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if cuda_available:
        # GPU 數量
        gpu_count = torch.cuda.device_count()
        print(f"GPU數量: {gpu_count}")
        
        # 顯示每個 GPU 的資訊
        for i in range(gpu_count):
            print(f"\nGPU {i}:")
            print(f"  名稱: {torch.cuda.get_device_name(i)}")
            
            # 記憶體資訊
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  總記憶體: {total_memory:.2f} GB")
            
            # 當前使用的記憶體
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                print(f"  已分配記憶體: {allocated:.2f} GB")
                print(f"  已保留記憶體: {reserved:.2f} GB")
        
        # 當前使用的 GPU
        current_device = torch.cuda.current_device()
        print(f"\n當前使用的GPU: {current_device}")
        
    else:
        print("\n⚠️ 注意：運行大型模型建議至少16GB VRAM，如無GPU可使用Google Colab")
    
    print("=" * 50)


if __name__ == "__main__":
    check_gpu_environment()