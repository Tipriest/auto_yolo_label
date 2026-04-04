## Setup GroundingDINO


### 1. steps
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