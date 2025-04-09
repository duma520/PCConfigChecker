# PC配置检测工具 - 人脸识别开发环境评估 说明书 v1.3

作者：杜玛
版权永久所有
日期：2025年
GitHub：https://github.com/duma520
网站：https://github.com/duma520

## 目录
1. [工具简介](#工具简介)
2. [适用人群](#适用人群)
3. [功能概述](#功能概述)
4. [安装指南](#安装指南)
5. [使用说明](#使用说明)
6. [检测内容详解](#检测内容详解)
7. [评估标准](#评估标准)
8. [常见问题](#常见问题)
9. [技术原理](#技术原理)
10. [开发者指南](#开发者指南)
11. [更新日志](#更新日志)

## 工具简介

PC配置检测工具是一款专为评估计算机是否适合进行人脸识别开发而设计的软件。它能全面检测您的硬件配置和软件环境，并给出针对人脸识别开发的适用性评分和建议。

**版本历史**：
- v1.0：基础硬件检测功能
- v1.1：增加AI框架检测
- v1.2：完善人脸识别库检测
- v1.3：增加详细评估和建议系统

## 适用人群

### 1. 完全不懂技术的小白
- **用途**：了解自己电脑的基本配置
- **能看懂**：硬件基本信息、总体评分和简单建议
- **举例**：知道自己的电脑内存是8GB还是16GB

### 2. 计算机爱好者
- **用途**：了解详细硬件参数和性能评估
- **能看懂**：CPU指令集、GPU显存等详细信息
- **举例**：了解自己的CPU是否支持AVX指令集

### 3. 人脸识别开发者
- **用途**：评估开发环境是否满足需求
- **能看懂**：AI框架支持情况、CUDA配置等
- **举例**：检查TensorFlow是否配置了GPU支持

### 4. 教育机构/培训机构
- **用途**：评估教学用机的配置是否达标
- **能看懂**：批量检测多台电脑的配置
- **举例**：检查机房电脑是否适合开展人脸识别课程

## 功能概述

### 主要功能模块
1. **系统信息检测**：操作系统、Python版本等
2. **硬件检测**：CPU、内存、磁盘、GPU
3. **AI框架检测**：TensorFlow、PyTorch等
4. **人脸识别库检测**：face_recognition、dlib等
5. **综合评估**：人脸识别开发适用性评分

### 界面说明
- **摘要**：关键信息概览
- **系统信息**：操作系统详情
- **硬件信息**：CPU、内存等详细信息
- **软件信息**：AI框架和人脸识别库
- **评估与建议**：评分和改进建议

## 安装指南

### 基础安装（适合小白）
1. 确保已安装Python 3.7或更高版本
2. 打开命令提示符（Windows）或终端（Mac/Linux）
3. 输入以下命令：
   ```
   pip install pc-config-checker
   ```
4. 安装完成后，输入以下命令运行：
   ```
   pc-config-checker
   ```

### 高级安装（适合开发者）
1. 克隆GitHub仓库：
   ```
   git clone https://github.com/duma520/pc-config-checker.git
   ```
2. 进入项目目录：
   ```
   cd pc-config-checker
   ```
3. 创建虚拟环境（推荐）：
   ```
   python -m venv venv
   ```
4. 激活虚拟环境：
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
5. 安装依赖：
   ```
   pip install -r requirements.txt
   ```
6. 运行程序：
   ```
   python pc_config_checker.py
   ```

## 使用说明

### 基本使用步骤
1. 启动程序后，点击"开始扫描"按钮
2. 等待扫描完成（通常需要10-30秒）
3. 查看各标签页的检测结果
4. 可点击"保存检测结果"将报告保存为JSON文件

### 界面操作指南
- **进度条**：显示扫描进度
- **状态标签**：显示当前扫描的项目
- **选项卡**：切换不同类别的检测结果
- **按钮功能**：
  - 开始扫描：启动检测过程
  - 保存结果：将报告保存到文件

## 检测内容详解

### 1. 系统信息
- 操作系统类型和版本
- 系统架构（32位/64位）
- 计算机名称
- Python版本

**专业说明**：这些信息对于确定软件兼容性非常重要，特别是Python版本会影响AI框架的选择。

### 2. CPU信息
- 品牌和型号
- 物理核心数和逻辑线程数
- 基础频率和当前频率
- 支持的指令集（如AVX）

**小白解释**：CPU就像电脑的大脑，核心数越多，处理能力越强。AVX是一种能加速计算的特殊指令。

### 3. 内存信息
- 总内存容量
- 可用内存
- 内存使用率

**举例说明**：8GB内存可以同时处理约1000张人脸检测，16GB则可处理更多。

### 4. 磁盘信息
- 各磁盘分区信息
- 文件系统类型
- 总容量和可用空间

**建议**：SSD硬盘比传统硬盘能更快加载人脸识别模型。

### 5. GPU信息
- GPU型号
- 显存容量
- GPU负载和温度
- 驱动版本

**专业说明**：NVIDIA GPU配合CUDA可以大幅加速深度学习运算。

### 6. AI框架支持
- TensorFlow/PyTorch是否安装
- 是否支持GPU加速
- CUDA/cuDNN配置

**常见问题**：如果显示"Protobuf版本问题"，可以尝试：
```
pip install protobuf==3.20.*
```

### 7. 人脸识别库
- face_recognition/dlib
- OpenCV人脸检测
- MTCNN/DeepFace等

**开发建议**：初学者可以从face_recognition库开始，专业开发推荐使用InsightFace。

## 评估标准

### 评分体系（满分100分）
1. **硬件评分（60%）**
   - CPU（25分）：核心数、频率、指令集
   - 内存（25分）：容量大小
   - GPU（25分）：有无独立GPU、显存大小
   - 磁盘（25分）：类型和剩余空间

2. **软件评分（40%）**
   - AI框架（50分）：TensorFlow/PyTorch等
   - 人脸识别库（50分）：face_recognition等

### 评分等级
- 80-100分：优秀，适合专业开发
- 60-79分：良好，适合学习和中小项目
- 40-59分：一般，基础功能可运行
- 0-39分：较差，建议升级配置

## 常见问题

### Q1: 扫描过程中卡住了怎么办？
A: 请等待1-2分钟，如果无响应，可以关闭程序重新启动。常见于首次检测GPU信息时。

### Q2: 为什么检测不到我的GPU？
A: 可能原因：
1. 使用的是集成显卡
2. 未安装NVIDIA驱动
3. 在虚拟机中运行

### Q3: 如何解决JAX初始化错误？
A: 尝试以下命令：
```
pip uninstall jax jaxlib -y
pip install --upgrade jax jaxlib
```

### Q4: NumPy版本冲突怎么办？
A: 执行以下命令降级NumPy：
```
pip install 'numpy<2'
```

## 技术原理

### 硬件检测实现
- 使用`cpuinfo`获取CPU详情
- 使用`psutil`获取内存和磁盘信息
- 使用`GPUtil`获取GPU数据

### 软件检测方法
- 尝试导入各AI框架库
- 检查CUDA/cuDNN是否存在
- 验证各人脸识别库是否可用

### 评估算法
```python
# 硬件评分计算示例
cpu_score = min(cpu_cores / 4 * 25, 25)  # 4核得25分
memory_score = min(memory_gb / 8 * 25, 25)  # 8GB得25分
gpu_score = 25 if has_gpu else 0

# 软件评分计算示例
framework_score = 0
if tf_or_torch: framework_score += 20
if opencv: framework_score += 10
if cuda_and_cudnn: framework_score += 20

# 总分计算
total_score = hardware_score * 0.6 + software_score * 0.4
```

## 开发者指南

### 代码结构
```
SystemScanner (QThread)
├── get_system_info()
├── get_cpu_info()
├── get_memory_info()
├── get_disk_info()
├── get_gpu_info()
├── check_ai_frameworks()
├── check_face_recognition_libs()
└── evaluate_for_face_recognition()

PCConfigChecker (QMainWindow)
├── init_ui()
├── start_scan()
├── update_progress()
├── scan_completed()
├── display_results()
└── save_results()
```

### 扩展检测项
如需添加新的检测项，可以在`SystemScanner`类中添加新方法，例如：

```python
def get_network_info(self):
    """检测网络信息"""
    import socket
    return {
        "主机名": socket.gethostname(),
        "IP地址": socket.gethostbyname(socket.gethostname())
    }
```

然后在`run()`方法中添加调用：
```python
# 在run()方法中添加（约第30行）
self.update_signal.emit("检测网络信息...", 35)
self.results['network_info'] = self.get_network_info()
```

## 更新日志

### v1.3 (当前版本)
- 增加详细的评估标准说明
- 优化JAX检测逻辑
- 添加更多错误处理

### v1.2
- 增加人脸识别库检测
- 完善评估建议系统
- 修复若干BUG

### v1.1
- 增加AI框架检测
- 添加GUI界面
- 支持结果保存

### v1.0
- 基础硬件检测功能
- 命令行界面

---

这份说明书旨在为不同技术水平的用户提供全面的使用指南。如果您有任何建议或发现问题，欢迎通过GitHub提交issue。我们将持续改进工具功能和文档质量。
