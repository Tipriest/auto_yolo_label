## 1. CVat安装
`CVat`具体就按照官方的[安装手册](https://docs.cvat.ai/docs/administration/community/basics/installation/)安装就好，它本身支持在线和`docker`两种模式，我使用docker进行安装的，docker容器一般你不主动关闭它，及时重启电脑等它也是不会关闭的，因此在安装好之后我可以一直使用`localhost:8080`来访问`CVat`的标注页面。

## 2. CVat使用方式简单介绍
我目前主要还是导入检测标注的数据集，因此一般就是这样，在打开CVat的后台页面之后，到`Projects`这个里面，来像下图一样创建一个新的项目
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat1.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>
创建新项目就写一个项目名就行，也不用写什么具体的标签啥的，因为一会导入的yolo的数据集都有这些
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat2.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="500"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>


然后在之前标注好的文件夹内，除了`visualizations`的东西不要，将其他的三个文件夹给统一压缩成一个`zip`文件

<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat4.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="700"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>

然后在`Projects`页选择`Import Dataset`
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat3.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>

在具体的设置项中，我一般会选择`Ultralytics YOLO Detection 1.0`数据集格式，然后上传刚才的压缩包
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat5.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="400"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>

然后打开这个`Projects`的详情页，就能够看到`train`和`val`数据集就在这个里面了，点击就可以进行标注了。
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat6.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>

标注的详情页大致就像下面这样
<div align="center" style="margin: 20px 0;">
    <img src="assets/cvat7.png"
        alt="A1_complex world"
        title="A1 Complex Gazebo World Environment"
        width="800"
        style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);"
        loading="lazy"/>
</div>