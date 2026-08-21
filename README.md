# AL-YOLO

**Real-Time Visual Target Detection and Tracking with YOLOv5**

基于 YOLOv5 的实时计算机视觉目标检测与追踪系统，面向游戏场景进行实时视觉分析。

项目最初开发于本科阶段，主要用于探索 **实时目标检测、GPU 推理、高速屏幕捕获以及低延迟视觉处理** 等技术。项目将 YOLOv5、GPU 推理、屏幕捕获和实时可视化等模块整合为一个完整的端到端视觉处理流程。

> **主要技术：** Python · PyTorch · YOLOv5 · CUDA · OpenCV · DXShot

## ✨ 项目特点

* 基于 **YOLOv5s** 的实时目标检测
* 使用 **DXShot** 进行高速屏幕捕获
* 基于 CUDA 的 GPU 推理
* 针对实时场景优化图像预处理与推理流程
* 支持实时目标检测结果可视化
* 提供轻量级 GUI 用于参数配置
* 支持多目标检测与目标优先级判断
* 支持不同机器环境下的参数自适应
* 在 NVIDIA RTX 3050 Laptop GPU 上达到约 **50 FPS** 的端到端处理速度

## 🏗️ 系统架构

```text
┌──────────────────┐
│  Screen Capture  │
│     DXShot       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Image Processing │
│     OpenCV       │
└────────┬─────────┘
         ↓
┌──────────────────┐
│   YOLOv5s Model  │
│   PyTorch/CUDA   │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Target Detection │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Visualization &  │
│ Control Interface│
└──────────────────┘
```

## ⚡ 性能优化

实时视觉系统中，屏幕捕获和模型推理都会直接影响整体处理延迟，因此项目对完整处理流程进行了针对性的性能优化。

最初使用 YOLOv5 自带的 MSS 进行屏幕捕获，单次截图耗时约 **15 ms**。经过性能测试后，将屏幕捕获模块替换为更高性能的 DXCam，并进一步使用 DXShot，将截图耗时降低至约 **5 ms**。

同时对 YOLOv5 推理流程进行了调整，移除部分不必要的框架层和处理步骤，并优化逐帧图像预处理，从而降低实时推理过程中的额外开销。

在测试环境下，完整处理流程可以达到约 **50 FPS**。

> 注意：由于系统通过屏幕捕获获取输入图像，因此最终 FPS 会受到显示器刷新率以及硬件性能的影响。

## 🧠 模型训练

项目使用 **YOLOv5s** 进行目标检测。

### 数据集

训练数据集：

[AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset)

### 训练方法

使用 [YOLOv5](https://github.com/ultralytics/yolov5) 提供的 `train.py` 进行模型训练，并根据数据集配置修改相应参数。

### 训练日志

![Training Log](https://user-images.githubusercontent.com/103171084/240091844-4bfd3af3-9d92-412e-8697-6d3ec8adc904.png)

训练日志以及实验过程中的相关数据保存在 `exp` 目录下。

## 🖥️ 屏幕捕获

项目对不同屏幕捕获方案进行了测试和比较。

最初使用 YOLOv5 默认的 MSS 进行截图，单次截图耗时约 **15 ms**。

随后尝试使用 DXCam，并进一步使用 DXShot：

```text
MSS
 ↓
~15 ms
 ↓
DXCam
 ↓
DXShot
 ↓
~5 ms
```

最终采用 DXShot 作为主要的屏幕捕获方案，以降低实时视觉处理中的输入延迟。

由于输入图像来自屏幕捕获，显示器刷新率也会限制系统能够达到的最大 FPS。

## 🎮 应用场景

项目最初以游戏环境作为实时计算机视觉的实验场景，通过游戏画面提供连续的视频流，用于研究：

* 实时目标检测
* 多目标识别
* 目标优先级判断
* 屏幕坐标与目标位置处理
* 低延迟视觉推理
* 实时控制接口

选择游戏环境的主要原因是其具有较高的实时性要求，可以作为测试目标检测模型和低延迟视觉处理方案的实际应用场景。

## 📦 环境依赖

默认运行环境为 Windows。

* Python >= 3.10, < 3.11
* CUDA 11
* PyTorch >= 2.0
* OpenCV
* 其他 Python 依赖见 `requirements.txt`

如果使用 GPU 进行推理，需要正确配置 CUDA、cuDNN 和 PyTorch。

## 🚀 快速开始

### 1. 创建 Python 环境

推荐使用 Conda：

```bash
conda create -n al-yolo python=3.10
conda activate al-yolo
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 CUDA

根据 NVIDIA GPU、CUDA 和 PyTorch 的版本要求完成 CUDA 环境配置。

### 4. 运行

```bash
python apex.py
```

具体配置请参考项目代码和相关配置文件。

## 📁 项目结构

```text
AL_Yolo/
├── configs/
├── exp/
├── models/
├── mouse_driver/
├── utils/
├── weights/
├── Capture.py
├── detect.py
├── apex.py
├── mouse_control.py
├── export.py
└── ...
```

## 🔧 后续改进

* [x] 优化屏幕捕获方式
* [ ] PID 平滑控制
* [ ] TensorRT 推理加速
* [x] 添加实时开关
* [x] 多目标识别与优先级判断
* [x] 项目架构优化
* [x] 不同机器参数自适应
* [ ] 推理部分 C++ 重写
* [ ] 数据集清洗与扩充
* [x] 实时可视化 GUI
* [ ] 生成安装包
* [ ] 支持更多 YOLO 系列模型

## 📈 Open Source

项目已开源，并获得 **350+ GitHub Stars**。

感谢所有对项目感兴趣并提供反馈的开发者。

## 🎉 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EthanH3514/AL_Yolo\&type=Date)](https://star-history.com/#EthanH3514/AL_Yolo&Date)

## 📄 License

See [LICENSE](LICENSE) for details.
