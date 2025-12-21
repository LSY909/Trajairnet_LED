#!/usr/bin/env python3
"""
根据原始数据目录生成 RAG 检索目录

使用方法:
    python build_rag_directory.py --source_dir ./dataset/7days3/processed_data --output_dir ./dataset/7days3_rag
    python build_rag_directory.py --source_dir ./dataset/7days3/processed_data --output_dir ./dataset/rag_7days3
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def build_rag_directory(source_dir: str, output_dir: str, include_splits: list = None):
    """
    从原始数据目录构建 RAG 检索目录
    
    Args:
        source_dir: 原始数据目录，应包含 train/ 和 test/ 子目录
        output_dir: 输出的 RAG 目录路径
        include_splits: 要包含的数据分割，如 ['train', 'test']，默认包含所有
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        raise ValueError(f"源目录不存在: {source_dir}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"创建输出目录: {output_dir}")
    
    # 确定要处理的子目录
    if include_splits is None:
        # 自动检测 train 和 test 目录
        splits = []
        for split in ['train', 'test']:
            split_path = source_path / split
            if split_path.exists() and split_path.is_dir():
                splits.append(split)
        if not splits:
            # 如果没有 train/test 子目录，直接处理源目录
            splits = ['.']
    else:
        splits = include_splits
    
    print(f"处理数据分割: {splits}")
    
    # 统计信息
    total_files = 0
    copied_files = 0
    skipped_files = 0
    
    # 遍历每个分割目录
    for split in splits:
        if split == '.':
            split_path = source_path
            split_name = 'root'
        else:
            split_path = source_path / split
            split_name = split
        
        if not split_path.exists():
            print(f"警告: 目录不存在，跳过: {split_path}")
            continue
        
        # 查找所有 .txt 文件
        txt_files = list(split_path.glob("*.txt"))
        total_files += len(txt_files)
        
        print(f"\n处理 {split_name} 目录: {len(txt_files)} 个文件")
        
        # 复制文件到输出目录
        for txt_file in tqdm(txt_files, desc=f"复制 {split_name} 文件"):
            output_file = output_path / txt_file.name
            
            # 如果文件已存在，添加前缀避免覆盖
            if output_file.exists():
                # 添加 split 前缀：train_1.txt 或 test_1.txt
                if split != '.':
                    new_name = f"{split}_{txt_file.name}"
                    output_file = output_path / new_name
                    if output_file.exists():
                        skipped_files += 1
                        continue
                else:
                    skipped_files += 1
                    continue
            
            # 复制文件
            shutil.copy2(txt_file, output_file)
            copied_files += 1
    
    print(f"\n完成!")
    print(f"  总文件数: {total_files}")
    print(f"  已复制: {copied_files}")
    print(f"  已跳过（重复）: {skipped_files}")
    print(f"  输出目录: {output_dir}")
    
    # 验证输出目录
    output_txt_files = list(output_path.glob("*.txt"))
    print(f"  输出目录中的文件数: {len(output_txt_files)}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="从原始数据目录生成 RAG 检索目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 7days3 生成 rag_7days3
  python build_rag_directory.py \\
      --source_dir ./dataset/7days3/processed_data \\
      --output_dir ./dataset/rag_7days3
  
  # 只包含 train 数据
  python build_rag_directory.py \\
      --source_dir ./dataset/7days3/processed_data \\
      --output_dir ./dataset/rag_7days3 \\
      --splits train
        """
    )
    
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
        help='原始数据目录路径（应包含 train/ 和 test/ 子目录）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出的 RAG 目录路径'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=None,
        choices=['train', 'test'],
        help='要包含的数据分割（默认包含所有可用的分割）'
    )
    
    args = parser.parse_args()
    
    build_rag_directory(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        include_splits=args.splits
    )


if __name__ == '__main__':
    main()

