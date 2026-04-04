## YOLO 自动标注流程
### 0. 环境配置
需要ROS1环境，中间的某几步需要用conda配置一个新环境
```shell
mkdir -p auto_yolo_label_ws/src && cd ./auto_yolo_label_ws
git clone git@github.com:Tipriest/auto_yolo_label.git ./src/auto_yolo_label
catkin build
```

### 0.1 最后期望的文件夹结构
```python

python src/auto_yolo_label/p01_generate_folder_structure/generate.py
```
```txt
--project_dir
    --p01_Videos
    --p02_Frames
    --p03_Grounding-DINO-First-Detection-Dataset
    --p04_CVat-Finetune-Dataset
    --p05_YOLO
    --p06_Yolo-Output-Dataset
    --p07_CVat-Finetune2-Dataset
    --p08_SAM-Dataset
    --p09_CVat-Finetune3-Dataset
    --(其他的可能的检测模型输出的分割模型)

```

### 1. ros消息转视频
修改配置文件`rostopic_to_video/config/rostopic_to_video.yaml`
- 这里现在有`RGB`, `Depth`相机深度视频的录制功能，深度的暂且不去管它
- 你只需要将`RGB`相机的话题消息给写到配置参数`rgb_topic`中就可以
- `output_dir`参数可以控制输出文件夹的位置，最好将保存位置设置到第一步中生成的`p01_Videos`文件夹中
- 默认名称将会保存名字为`rgb.mp4`和`depth.mp4`的两个视频

```shell
source devel/setup.bash
roslaunch rostopic_to_video rostopic_to_video.launch
```

### 2. 视频帧提取
修改配置文件`rostopic_to_video/config/rostopic_to_video.yaml`
- 需要修改输入的需要提取帧的视频的位置`input_video`
- 可以修改保存输出的视频帧的位置`frames`，最好将保存位置设置到第一步中生成的`p02_Frames`文件夹中
- 可以修改其他关于帧选择的策略等
- 可以选择其他关于保存的帧的格式等

```shell
source devel/setup.bash
roslaunch video_frame_extractor video_frame_extractor.launch
```

### 3. grounding-dino自动标注
这个需要配置一下grounding-dino的环境，具体的配置过程参考这个[教程](./setup_groundingdino.md)

修改配置文件`src/auto_yolo_label/third_party/GroundingDINO/groundingdino_to_yolo.yaml`
- 改一下类别名
- 改一下输入文件夹，建议的输入文件夹为第二步中生成的`p02_Frames`文件夹
- 改一下输出文件夹，建议的输出文件夹为第一步中生成的`p03_Grounding-DINO-First-Detection-Dataset`文件夹

```shell
cd src/auto_yolo_label/third_party/GroundingDINO
python groundingdino_to_yolo.py
```

大致瞅了一眼，识别效果还行,但是比较小的石头的识别效果就比较差一点了
<div align="center" style="margin: 20px 0;">
    <img src="assets/frame_000020.jpg"
        alt="mars stone grounding dino recognize"
        title="mars stone grounding dino recognize"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>


### 4. CVat手工补标
有很多做标注的工具，我之前用过像是`Label-studio`，`CVat`这些主要是标注的工具，像是`fifty-one`是用来做数据分析的工具，我认为对于一个标注工具是否好用主要在于:
- 导入格式支持(能不能把常见的格式压缩一个压缩包就导入)
- 导出格式支持
- 能不能调用一些自己的后端进行粗标，比如后端接一个标注的模型等
像是`Label-studio`我觉得它就不擅长从已经标注好的数据集进行导入，但是它本身支持的后端还好一些，但是后来我就不用这个了，转而是用`CVat`，对导入的支持比较好些。
关于CVat的安装和使用，具体可以参考这个[教程](./setup_cvat.md)
- 导出的数据集的路径设置建议，可以设置到第一步中生成的`p04_CVat-Finetune-Dataset`文件夹中
- 关于导出这里有一个细节需要注意一下，即使导出的时候需要选择`Ultralytics YOLO Detection 1.0`这个格式，CVat导出的数据集的文件夹结构和yolo训练需要的文件夹结构是仍然是有一些差别的，具体的修改的办法是将导出的`val.txt`,`train.txt`, `test.txt`和`data.yaml`删除掉，换上第3步的时候生成的`dataset.yaml就可以了`


### 5. yolo训练
`Yolo`训练的部分现在还特别粗糙，有很多增强训练的办法还有待继续加上，这一部分需要使用第3节中安装的`grounding-dino`环境
```shell
conda activate grounding-dino
pip install ultralytics

cd <your-workspace>/src/auto_yolo_label/yolo_train
mkdir dataset && cd dataset
# 将你从CVat上下载好的数据集压缩包解压至dataset文件夹
# 更改train.py中的相关的地方，主要是更改一下用什么模型，使用的数据集的yaml文件的位置
# 最后的onnx可以导出也可以不导出，如果想要纯用cpu推理的话就导出就行

# 使用python train.py进行训练
python train.py

# 使用infer_video.py进行视频的推理
python infer_video.py
```
