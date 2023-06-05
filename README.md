### 一个基于yolov5的Apex Legend AI辅瞄外挂
## 环境依赖
* 罗技驱动
* python3.10
* CUDA 11
* opencv
## 项目流程图
![项目流程图](https://user-images.githubusercontent.com/103171084/240091322-a9e21a41-faa8-4ce8-96b5-0be957c86012.png)
### 模型训练
#### 数据集地址
[https://github.com/goldjee/AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset)
#### 训练方法
使用[Yolov5](https://github.com/ultralytics/yolov5) 的train.py文件，修改数据集路径参数
#### 训练日志
![log](https://user-images.githubusercontent.com/103171084/240091844-4bfd3af3-9d92-412e-8697-6d3ec8adc904.png)
### 屏幕捕捉
尝试手写一个Capturer类，但是效率不高

转为使用yolov5自带的LoadScreenShot函数，修改默认参数(传入图像尺寸)

同时通过多开一个线程给监视器来监视全局变量的变化来跨文件传参

### 鼠标控制
#### 如何控制鼠标
大部分FPS游戏将win函数屏蔽了，所以转为操纵鼠标驱动来模拟鼠标输入。

操纵罗技鼠标驱动的文件之前有人写过，就是这个代码中的logitech.driver.dll

#### 鼠标移动函数
![](https://user-images.githubusercontent.com/103171084/241761731-2293a5f2-6421-4d37-b353-d6ec7ea2ccc7.png)

暂时没想出来如何将鼠标直接移过去，后面会改

目前采用的是写一个几步之内收敛到目标框内的函数，牺牲了很大的效率，但也能达到目的

### 开/关 鼠标控制模块
将`mouse_control.py`中的`driver.moveR(int(x), int(y), True)`注释掉即可
### 后续改进

- [ ] 一帧拉枪
- [ ] 多目标识别优先级判断
- [x] 项目架构优化
- [ ] 对不同机器参数自适应
- [ ] 推理部分C++重写
- [ ] 数据集清洗，扩充
- [ ] 做个前端
- [ ] 生成安装包