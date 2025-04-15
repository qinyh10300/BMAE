import os
import shutil
import pandas as pd
from tqdm import tqdm

def create_imagefolder_structure(csv_file, img_dir, output_dir, split_name):
    """
    根据 CSV 文件，将图像分类并保存为 ImageFolder 格式的目录结构
    :param csv_file: CSV 文件路径，包含图像名称和类标签
    :param img_dir: 原始图像所在目录
    :param output_dir: 输出目标目录（将生成 train, val, test 文件夹）
    :param split_name: 切分名称（train、val、test）
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 为每个分割创建目录
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # 遍历 CSV 中的每一行，将图像移动到相应的文件夹
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        img_name = row['filename']  # 假设 CSV 文件中有 "image" 列
        class_label = row['label']  # 假设 CSV 文件中有 "label" 列
        
        # 构造类别目录路径
        class_dir = os.path.join(split_dir, str(class_label))
        os.makedirs(class_dir, exist_ok=True)
        
        # 构造源文件和目标文件路径
        src_path = os.path.join(img_dir, img_name)
        dst_path = os.path.join(class_dir, img_name)
        
        # 移动图像文件到新位置
        shutil.copy(src_path, dst_path)

def organize_dataset(csv_files, img_dir, output_dir):
    """
    根据提供的 CSV 文件和图像目录，整理数据集
    :param csv_files: 包含 train.csv、val.csv、test.csv 的字典
    :param img_dir: 原始图像目录
    :param output_dir: 输出目录
    """
    for split_name, csv_file in csv_files.items():
        print(f"Processing {split_name} split...")
        create_imagefolder_structure(csv_file, img_dir, output_dir, split_name)
    print("Dataset organized successfully!")

if __name__ == "__main__":
    # 输入路径
    #
    base_dir = "/home/qinyh/.cache/kagglehub/datasets/hylanj/mini-imagenetformat-csv/versions/1"
    images_dir = base_dir + "/images/images"  # 图像目录
    output_dir = "./datasets/mini_imagenet"  # 输出目标目录

    # CSV 文件路径，假设 CSV 文件有 "image" 列和 "label" 列
    csv_files = {
        'train': base_dir + '/train.csv',  # 训练集
        'val': base_dir + '/val.csv',      # 验证集
        'test': base_dir + '/test.csv'     # 测试集
    }

    # 整理数据集
    organize_dataset(csv_files, images_dir, output_dir)
