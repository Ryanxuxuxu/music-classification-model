"""
将PyTorch模型checkpoint文件拆分成多个小文件的工具脚本。
每个分片文件不超过指定大小（默认15MB）。
"""
import argparse
import os
import pickle
import torch
from typing import List


def split_checkpoint(checkpoint_path: str, output_dir: str = None, max_size_mb: int = 15):
    """
    将checkpoint文件拆分成多个小文件。
    
    Args:
        checkpoint_path: 原始checkpoint文件路径
        output_dir: 输出目录，如果为None则使用checkpoint文件所在目录
        max_size_mb: 每个分片文件的最大大小（MB）
    
    Returns:
        分片文件路径列表
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始checkpoint
    print(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 将checkpoint序列化为字节
    print("正在序列化checkpoint...")
    checkpoint_bytes = pickle.dumps(checkpoint)
    total_size = len(checkpoint_bytes)
    total_size_mb = total_size / (1024 * 1024)
    print(f"Checkpoint总大小: {total_size_mb:.2f} MB")
    
    # 计算分片大小（字节）
    max_size_bytes = max_size_mb * 1024 * 1024
    num_chunks = (total_size + max_size_bytes - 1) // max_size_bytes  # 向上取整
    
    print(f"将拆分为 {num_chunks} 个文件，每个不超过 {max_size_mb} MB")
    
    # 生成基础文件名
    base_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    chunk_files = []
    
    # 拆分并保存
    for i in range(num_chunks):
        start = i * max_size_bytes
        end = min(start + max_size_bytes, total_size)
        chunk = checkpoint_bytes[start:end]
        
        chunk_filename = f"{base_name}.part{i+1:03d}.pth"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        # 保存分片（包含元数据）
        chunk_data = {
            'chunk_index': i,
            'total_chunks': num_chunks,
            'chunk_size': len(chunk),
            'total_size': total_size,
            'data': chunk
        }
        
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        
        chunk_size_mb = len(chunk) / (1024 * 1024)
        print(f"  已保存分片 {i+1}/{num_chunks}: {chunk_filename} ({chunk_size_mb:.2f} MB)")
        chunk_files.append(chunk_path)
    
    # 保存元数据文件
    metadata = {
        'original_file': os.path.basename(checkpoint_path),
        'total_chunks': num_chunks,
        'total_size': total_size,
        'chunk_files': [os.path.basename(f) for f in chunk_files]
    }
    metadata_path = os.path.join(output_dir, f"{base_name}.meta.json")
    
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n拆分完成！")
    print(f"元数据文件: {metadata_path}")
    print(f"分片文件数量: {num_chunks}")
    
    return chunk_files


def merge_checkpoint_chunks(chunk_files: List[str] = None, base_path: str = None, device: str = None):
    """
    合并拆分后的checkpoint文件。
    
    Args:
        chunk_files: 分片文件路径列表，如果为None则从base_path自动检测
        base_path: checkpoint的基础路径（不含.partXXX.pth后缀），用于自动查找分片
        device: 加载设备
    
    Returns:
        合并后的checkpoint字典
    """
    if chunk_files is None:
        if base_path is None:
            raise ValueError("必须提供chunk_files或base_path")
        
        # 自动查找所有分片文件
        base_dir = os.path.dirname(base_path) if os.path.dirname(base_path) else '.'
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        # 查找所有匹配的分片文件
        chunk_files = []
        i = 1
        while True:
            chunk_path = os.path.join(base_dir, f"{base_name}.part{i:03d}.pth")
            if not os.path.exists(chunk_path):
                break
            chunk_files.append(chunk_path)
            i += 1
        
        if not chunk_files:
            raise FileNotFoundError(f"未找到分片文件: {base_path}.part*.pth")
    
    # 按索引排序
    chunk_files = sorted(chunk_files)
    
    print(f"正在合并 {len(chunk_files)} 个分片文件...")
    
    # 读取所有分片
    chunks_data = []
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            chunks_data.append(chunk_data)
    
    # 验证分片完整性
    total_chunks = chunks_data[0]['total_chunks']
    total_size = chunks_data[0]['total_size']
    
    if len(chunks_data) != total_chunks:
        raise ValueError(f"分片数量不匹配: 期望 {total_chunks}, 实际 {len(chunks_data)}")
    
    # 合并所有分片数据
    checkpoint_bytes = b''.join([chunk['data'] for chunk in chunks_data])
    
    if len(checkpoint_bytes) != total_size:
        raise ValueError(f"合并后大小不匹配: 期望 {total_size}, 实际 {len(checkpoint_bytes)}")
    
    # 反序列化checkpoint
    print("正在反序列化checkpoint...")
    checkpoint = pickle.loads(checkpoint_bytes)
    
    # 如果指定了device，将tensor移到对应设备
    if device is not None:
        if 'model_state_dict' in checkpoint:
            checkpoint['model_state_dict'] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in checkpoint['model_state_dict'].items()
            }
    
    print("合并完成！")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="拆分或合并PyTorch checkpoint文件")
    parser.add_argument("--mode", choices=['split', 'merge'], required=True,
                       help="操作模式: split=拆分, merge=合并")
    parser.add_argument("--input", required=True,
                       help="输入文件路径（拆分模式：原始checkpoint；合并模式：基础路径或分片文件列表）")
    parser.add_argument("--output_dir", default=None,
                       help="输出目录（仅拆分模式）")
    parser.add_argument("--max_size_mb", type=int, default=15,
                       help="每个分片的最大大小（MB），默认15MB")
    
    args = parser.parse_args()
    
    if args.mode == 'split':
        split_checkpoint(args.input, args.output_dir, args.max_size_mb)
    elif args.mode == 'merge':
        # 尝试作为基础路径处理
        if os.path.exists(args.input):
            # 如果是文件，尝试作为第一个分片
            checkpoint = merge_checkpoint_chunks(base_path=args.input)
        else:
            # 作为基础路径处理
            checkpoint = merge_checkpoint_chunks(base_path=args.input)
        
        # 保存合并后的文件
        output_path = args.input.replace('.part', '_merged').replace('.pth', '_merged.pth')
        if not output_path.endswith('.pth'):
            output_path = args.input + '_merged.pth'
        torch.save(checkpoint, output_path)
        print(f"合并后的checkpoint已保存到: {output_path}")


if __name__ == "__main__":
    main()

