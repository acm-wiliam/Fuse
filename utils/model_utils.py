import os
import torch
import json
import time

def save_model(model, path, metadata=None):
    """
    保存PyTorch模型和相关元数据
    
    参数:
        model: 要保存的模型
        path: 保存路径（目录）
        metadata: 要保存的元数据字典（可选）
    """
    # 创建目录（如果不存在）
    os.makedirs(path, exist_ok=True)
    
    # 构建文件路径
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(path, f'model_{timestamp}.pth')
    
    # 保存模型
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存元数据（如果提供）
    if metadata is not None:
        metadata_path = os.path.join(path, f'metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"元数据已保存到: {metadata_path}")

    return model_path

def load_model(model, path):
    """
    加载PyTorch模型
    
    参数:
        model: 要加载权重的空模型实例
        path: 模型权重的文件路径
        
    返回:
        加载权重后的模型
    """
    try:
        # 加载模型权重
        model.load_state_dict(torch.load(path))
        print(f"模型已从 {path} 加载")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def get_device():
    """
    获取可用的计算设备
    
    返回:
        torch.device: 'cuda' 如果 CUDA 可用，否则为 'cpu'
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')