UnrealSynth 虚幻合成数据生成器利用虚幻引擎的实时渲染能力搭建逼真的三维场景，为 YOLO 等 AI 模型的训练提供自动生成的图像和标注数据，官方下载地址：[UnrealSynth](https://tools.nsdt.cloud/UnrealSynth)。
UnrealSynth 生成的合成数据可用于深度学习模型的训练和验证，可以极大地提高各种行业细分场景中目标识别任务的实施效率，例如：安全帽检测、交通标志检测、施工机械检测、车辆检测、行人检测、船舶检测等。

![image](https://github.com/ciga2011/UnrealSynth/assets/3120837/67e9c243-f6e9-4ec6-8e50-821290d17652)


## 1、UnrealSynth 合成数据工具包内容
UnrealSynth 基于 UE5 虚幻引擎开发，目前支持 YOLO 系列模型合成数据的生成，当前版本号 V1.0，主要文件和目录的组织结构如下：

|目录|	内容|	
|-|-|
|Engine/|	发布本程序的原始软件的编码和资源文件，其中包含构件此程序的二进制编码和一些存放在 content 文件中的原始资产等	|
|UnrealSynth/Binaries/|	本程序兼容系统及其他的二进制文件|	
|UnrealSynth/Content/|	本程序中所使用的所有资产文件已被烘焙成 pak 包|	
|UnrealSynth.exe|	运行程序|	
|LICENSE.md|	开发包许可协议文件|

运行UnrealSynth的推荐配置为：

- 处理器：13th Gen Intel(R) Core(TM) i5-13400 2.50 GHz
- RAM：64.0 GB
- 独显：NVIDIA GeForce RTX 3080 Ti

## 2、UnrealSynth 合成数据生成
以下是以 YOLO 模型为例，详细讲述如何使用 UnrealSynth 虚幻引擎数据生成器来生成为 YOLO 模型生成训练的合成数据。

打开 UnrealSynth 虚幻引擎合成数据生成器，点击【虚幻合成数据生成器】按钮，进入虚幻场景编辑页面，点击【环境变更】按钮切换合适的场景，输入【模型类别】参数后就可以开始导入模型，点击【导入 GLB 模型】弹出文件选择框，任意选择一个 GLB 文件，这里以抱枕文件为例，添加抱枕 GLB 文件后的场景如下：

![image](https://github.com/ciga2011/UnrealSynth/assets/3120837/ae8afc38-9b34-4387-a913-ed5557506f8b)


将 GLB 文件添加到场景后，接下来就可以配置 UnrealSynth 合成数据生成参数，参数配置说明如下：

- 模型类别: 生成合成数据 synth.yaml 文件中记录物体的类型
- 环境变更 : 变更场景背景
- 截图数量 : 生成合成数据集 image 目录下的图像数量，在 train 和 val 目录下各自生成总数一半数量的图片
- 物体个数 : 设置场景中的物体个数，目前最多支持 5 个，并且是随机的选取模型的类别
- 随机旋转 : 场景中的物体随机旋转角度
- 随机高度 : 场景中的物体随机移动的高度
- 截图分辨率: 生成的 images 图像数据集中的图像分辨率
- 缩放 : 物体缩放调整大小

点击【确定】后会在本地目录中...\UnrealSynth\Windows\UnrealSynth\Content\UserData 自动生成两个文件夹以及一个 yaml 文件：images、labels、test.yaml 文件。

```
UnrealSynth\Windows\UnrealSynth\Content\UserData
    |- images
        |-train
            |- 0.png
            |- 1.png
            |- 2.png
            |- ...
         |-val
            |- 0.png
            |- 1.png
            |- 2.png
            |- ...
    |- labels
        |-train
            |- 0.txt
            |- 1.txt
            |- 2.txt
            |- ...
        |-val
            |- 0.txt
            |- 1.txt
            |- 2.txt
            |- ...
    |- synth.yaml
```

UnrealSynth 合成数据已生成，可以利用数据集训练 YOLO 模型，会在 images 下生成两个图像目录：train 和 val。train 目录表示训练图像数据目录，val 表示验证图像数据目录。

例如 train 目录下的图像集合：

![image](https://github.com/ciga2011/UnrealSynth/assets/3120837/4708f713-6840-4df5-ac56-4e1660ac7721)


同样在 labels 标注目录下也会生成两个标注目录：train 和 val。

train 目录表示标注训练数据目录，val 表示标注验证数据目录。

生成的 labels 标注数据格式如下：
```
0 0.68724 0.458796 0.024479 0.039815
0 0.511719 0.504167 0.021354 0.034259
0 0.550781 0.596759 0.039062 0.04537
0 0.549219 0.368519 0.023438 0.044444
0 0.47526 0.504167 0.009896 0.030556
0 0.470313 0.69537 0.027083 0.035185
0 0.570052 0.499074 0.016146 0.040741
0 0.413542 0.344444 0.022917 0.037037
0 0.613802 0.562037 0.015104 0.027778
0 0.477344 0.569444 0.017188 0.016667
```

生成的 synth.yaml 数据格式如下：
```
path:
train: images
val: images
test:
names:
 0: pear
 1: Fruit tray
 2: apple
 3: papaya
 4: apple
```

## 3、利用 UnrealSynth 合成数据训练 YOLOv8 模型
数据集生成后有三个办法可以进行模型训练：使用 python 脚本、使用命令行、使用在线服务。

第一种是使用 python 脚本,需首先安装 ultralytics 包，训练代码如下所示：
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='synth.yaml', epochs=100, imgsz=640)
```

第二种是使用命令行，需安装 YOLO 命令行工具，训练代码如下：
```
# Build a new model from YAML and start training from scratch
yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

# Start training from a pretrained *.pt model
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

# Build a new model from YAML, transfer pretrained weights to it and start training
yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
```

第三种是使用ultralytics hub 或者其他在线训练工具。
