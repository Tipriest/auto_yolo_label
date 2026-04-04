import os
from datetime import datetime
import time

if __name__ == "__main__":
    # 定义文件夹结构
    base_folder = "/data/self_make"
    project_name = "water_world"
    project_name_timesuffix = project_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folders = [
        f"{base_folder}/{project_name_timesuffix}/p01_Videos",
        f"{base_folder}/{project_name_timesuffix}/p02_Frames",
        f"{base_folder}/{project_name_timesuffix}/p03_Grounding-DINO-First-Detection-Dataset",
        f"{base_folder}/{project_name_timesuffix}/p04_CVat-Finetune-Dataset",
        f"{base_folder}/{project_name_timesuffix}/p05_YOLO",
        f"{base_folder}/{project_name_timesuffix}/p06_Yolo-Output-Dataset",
        f"{base_folder}/{project_name_timesuffix}/p07_CVat-Finetune2-Dataset",
        f"{base_folder}/{project_name_timesuffix}/p08_SAM-Dataset",
        f"{base_folder}/{project_name_timesuffix}/p09_CVat-Finetune3-Dataset",
    ]

    # 创建文件夹
    for folder in folders:
        if os.path.exists(folder):
            os.rmdir(folder)  # 删除已存在的文件夹
        os.makedirs(folder)

    print("文件夹结构已生成：")
    for folder in folders:
        print(f" - {folder}")