## YOLO 自动标注流程
### 0. 环境配置
需要ROS1环境，中间的某几步需要用conda配置一个新环境
```shell
mkdir -p auto_yolo_label_ws/src && cd ./auto_yolo_label_ws
git clone git@github.com:Tipriest/auto_yolo_label.git ./src/auto_yolo_label
catkin build
```

### 1. ros消息转视频
修改配置文件`rostopic_to_video/config/rostopic_to_video.yaml`
- 这里现在有`RGB`, `Depth`相机深度视频的录制功能，深度的暂且不去管它
- 你只需要将`RGB`相机的话题消息给写到配置参数`rgb_topic`中就可以
- `output_dir`参数可以控制输出文件夹的位置，默认保存位置是`rostopic_to_video`下面叫做`videos`的文件夹内
- 默认名称将会保存名字为`rgb.mp4`和`depth.mp4`的两个视频

```shell
source devel/setup.bash
roslaunch rostopic_to_video rostopic_to_video.launch
```

### 2. 视频帧提取
修改配置文件`rostopic_to_video/config/rostopic_to_video.yaml`
- 需要修改输入的需要提取帧的视频的位置`input_video`
- 可以修改保存输出的视频帧的位置`frames`
- 可以修改其他关于帧选择的策略等
- 可以选择其他关于保存的帧的格式等

```shell
source devel/setup.bash
roslaunch rostopic_to_video rostopic_to_video.launch
```

### 3. grounding-dino自动标注
这个需要配置一下grounding-dino的环境
```shell
git submodule update --init --recursive
conda create -n grounding-dino python==3.9
conda activate grounding-dino
cd ./third_party/GroundingDINO
```
需要注意的是，安装`grounding-dino`需要`pytorch`，安装`pytorch`需要与自己的`cuda`版本相吻合，我一般使用`update-alternatives`来管理我使用的`cuda`版本，如该[链接](https://tipriest.blog.csdn.net/article/details/149880758?spm=1011.2415.3001.5331)所示，读者如果需要可以自行取用

在确定好`cuda`版本之后,请读者在在这个[网站](https://pytorch.org/get-started/previous-versions/)根据自己的cuda版本来安装需要的`torch`，我使用`cuda12.4`的版本，因此我的安装命令为
```shell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install pip==22.3.1
pip install "setuptools>=62.3.0,<75.9"
pip install --no-build-isolation -e .

# 下载权重
mkdir weights
cd weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```
grounding dino开放的有两个权重，还可以在[这个网址](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha2)单独去下载grounding-dino更大的一个权重，下载好之后也是放到`weights`文件夹中
在配置好grounding-dino环境后，可以在其目录下运行`test.py`来验证是否已经完成安装并且工作正常

修改配置文件`rostopic_to_video/config/rostopic_to_video.yaml`
- 改一下类别名
- 改一下输入输出文件夹

大致瞅了一眼，识别效果还行,但是比较小的石头的识别效果就比较差一点了
<div align="center" style="margin: 20px 0;">
    <img src="assets/frame_000020.jpg"
        alt="mars stone grounding dino recognize"
        title="mars stone grounding dino recognize"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>


### 4. cvat手工补标


### 5. yolo训练

