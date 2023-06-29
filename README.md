### 一个基于yolov5的Apex Legend AI辅瞄外挂
## 环境依赖
* 罗技驱动(版本不超过21.9)
* python3.10
* CUDA 11
* opencv
* torch 2.0
## 项目流程图
![项目流程图](https://user-images.githubusercontent.com/103171084/240091322-a9e21a41-faa8-4ce8-96b5-0be957c86012.png)
### 模型训练
#### 数据集地址
[https://github.com/goldjee/AL-YOLO-dataset](https://github.com/goldjee/AL-YOLO-dataset)
#### 训练方法
使用[Yolov5](https://github.com/ultralytics/yolov5) 的train.py文件，修改数据集路径参数
#### 训练日志
![log](https://user-images.githubusercontent.com/103171084/240091844-4bfd3af3-9d92-412e-8697-6d3ec8adc904.png)

训练日志以及过程的数据存放在`exp`目录下
### 屏幕捕捉
~~尝试手写一个Capturer类，但是效率不高~~

~~转为使用yolov5自带的`LoadScreenShot`函数，修改默认参数(传入图像尺寸)~~

~~使用[DXcam](https://github.com/ra1nty/DXcam)截图来替代yolov5自带的mss截图，将截图时间从15ms优化到5ms~~

使用[Dxshot](https://github.com/AI-M-BOT/DXcam/releases)将截图时间进一步优化

同时通过多开一个线程给监视器来监视全局变量的变化来跨文件传参

注意：因为是通过屏幕截图的方式来获取的图像，所以屏幕刷新率会限制FPS的上限

### 鼠标控制
#### 如何控制鼠标
~~大部分FPS游戏将win函数屏蔽了，所以转为操纵鼠标驱动来模拟鼠标输入。~~
Apex并未屏蔽win函数，仍然可以使用win函数控制鼠标移动，后续会加入这个选项，目前暂时还是使用鼠标驱动。

操纵罗技鼠标驱动的文件之前有人写过，就是`mouse_driver`中的`ghub_mouse.dll`

#### 鼠标移动函数
![](https://user-images.githubusercontent.com/103171084/241761731-2293a5f2-6421-4d37-b353-d6ec7ea2ccc7.png)

暂时没想出来如何将鼠标直接移过去，网上也找不到前人的经验。

基于内存的外挂自瞄原理是可以拿到三维坐标，直接修改方向角来瞄准敌人，而基于计算机视觉的外挂只能拿到目标在屏幕上的投影，这是一个二维坐标，要解算出移动的向量很依赖游戏底层的参数(视场角等，详见[issue](https://github.com/EthanH3514/AL_Yolo/issues/3))，目前还没想明白怎么一帧锁敌，也许将来会去实现。

况且一帧锁敌会大大提高被检测的风险性，实现的价值不大，短期内不会再向这个方向去努力了。

目前采用的是将准星逐步移动到目标身上的方法，牺牲了一点点效率，达到了准星吸附一样的效果。

### 如何使用
- 将代码下载到本地
- 部署环境依赖
- 修改参数(目前在`mouse_control.py`下)
- 管理员模式打开一个终端，进入项目文件夹下运行`python apex.py`
- 退出目标检测：对截图窗口输入`q`
- 程序退出：按下`End`键

### 后续改进

- [x] 截图方式优化
- [ ] 推理文件多线程并行
- [ ] 加入PID平滑控制鼠标
- [ ] tensorrt推理加速
- [ ] 添加自瞄开关
- [ ] 取消对驱动的依赖
- [ ] 多目标识别优先级判断
- [x] 项目架构优化
- [x] 对不同机器参数自适应
- [ ] 推理部分C++重写
- [ ] 数据集清洗，扩充，加入敌我识别
- [ ] 做个前端
- [ ] 生成安装包
- [ ] 一帧拉枪(太难，与内存挂原理不同)


### 如果有帮到你就点一个star吧?