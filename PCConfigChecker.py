# PC配置检测工具 v1.2 - 人脸识别开发环境评估
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 禁用TensorFlow日志输出
import sys
import platform
import cpuinfo
import psutil
import GPUtil
import subprocess
import json
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QTabWidget, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
CHECK_JAX = False
import numpy as np
if np.__version__.startswith('2.'):
    print("警告: NumPy 2.x 可能与某些库不兼容，建议降级到 NumPy 1.x")
    print("执行: pip install 'numpy<2'")

class SystemScanner(QThread):
    update_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.results = {}

    def run(self):
        try:
            self.update_signal.emit("开始系统扫描...", 0)
            
            # 1. 系统基本信息
            self.update_signal.emit("收集系统基本信息...", 10)
            self.results['system_info'] = self.get_system_info()
            
            # 2. CPU信息
            self.update_signal.emit("检测CPU信息...", 20)
            self.results['cpu_info'] = self.get_cpu_info()
            
            # 3. 内存信息
            self.update_signal.emit("检测内存信息...", 30)
            self.results['memory_info'] = self.get_memory_info()
            
            # 4. 磁盘信息
            self.update_signal.emit("检测磁盘信息...", 40)
            self.results['disk_info'] = self.get_disk_info()
            
            # 5. GPU信息
            self.update_signal.emit("检测GPU信息...", 50)
            self.results['gpu_info'] = self.get_gpu_info()
            
            # 6. AI框架支持检测
            self.update_signal.emit("检测AI框架支持...", 60)
            self.results['ai_frameworks'] = self.check_ai_frameworks()
            
            # 7. 人脸识别库检测
            self.update_signal.emit("检测人脸识别库...", 80)
            self.results['face_recognition_libs'] = self.check_face_recognition_libs()
            
            # 8. 评估人脸识别开发适用性
            self.update_signal.emit("评估人脸识别开发适用性...", 90)
            self.results['evaluation'] = self.evaluate_for_face_recognition()
            
            self.update_signal.emit("扫描完成!", 100)
            self.finished_signal.emit(self.results)
            
        except Exception as e:
            error_msg = f"扫描过程中出错: {str(e)}"
            if "jax" in str(e).lower():
                error_msg += "\nJAX库问题建议解决方案:"
                error_msg += "\n1. pip uninstall jax jaxlib -y"
                error_msg += "\n2. pip install --upgrade jax jaxlib"
            print(error_msg, file=sys.stderr)
            self.update_signal.emit(error_msg, 0)

            

    def get_system_info(self):
        info = {
            "系统类型": platform.system(),
            "系统版本": platform.version(),
            "系统架构": platform.architecture()[0],
            "计算机名称": platform.node(),
            "Python版本": platform.python_version(),
            "处理器名称": platform.processor()
        }
        return info

    def get_cpu_info(self):
        info = cpuinfo.get_cpu_info()
        cpu_data = {
            "品牌": info.get('brand_raw', '未知'),
            "架构": info.get('arch', '未知'),
            "位数": info.get('bits', '未知'),
            "核心数(物理)": psutil.cpu_count(logical=False),
            "线程数(逻辑)": psutil.cpu_count(logical=True),
            "基础频率": info.get('hz_advertised_friendly', '未知'),
            "当前频率": info.get('hz_actual_friendly', '未知'),
            "支持指令集": info.get('flags', '未知')
        }
        return cpu_data

    def get_memory_info(self):
        mem = psutil.virtual_memory()
        return {
            "总内存(GB)": round(mem.total / (1024**3), 2),
            "可用内存(GB)": round(mem.available / (1024**3), 2),
            "已用内存(GB)": round(mem.used / (1024**3), 2),
            "内存使用率(%)": mem.percent
        }

    def get_disk_info(self):
        disks = []
        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            disks.append({
                "设备": part.device,
                "挂载点": part.mountpoint,
                "文件系统": part.fstype,
                "总容量(GB)": round(usage.total / (1024**3), 2),
                "已用(GB)": round(usage.used / (1024**3), 2),
                "可用(GB)": round(usage.free / (1024**3), 2),
                "使用率(%)": usage.percent
            })
        return disks

    def get_gpu_info(self):
        gpus = []
        try:
            for gpu in GPUtil.getGPUs():
                gpus.append({
                    "ID": gpu.id,
                    "名称": gpu.name,
                    "显存(GB)": f"{gpu.memoryTotal:.1f}",
                    "已用显存(GB)": f"{gpu.memoryUsed:.1f}",
                    "显存使用率(%)": f"{gpu.memoryUtil * 100:.1f}",
                    "GPU负载(%)": f"{gpu.load * 100:.1f}",
                    "温度(°C)": f"{gpu.temperature:.1f}",
                    "驱动版本": gpu.driver if hasattr(gpu, 'driver') else '未知'
                })
        except Exception as e:
            gpus.append({"错误": f"无法获取GPU信息: {str(e)}"})
        return gpus

    def check_ai_frameworks(self):
        frameworks = {
            "TensorFlow": False,
            "PyTorch": False,
            "Keras": False,
            "OpenCV": False,
            "ONNX Runtime": False,
            "CUDA": False,
            "cuDNN": False,
            "Protobuf 版本问题": False,
            "JAX": False
        }
        
        # 检查TensorFlow
        try:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            import tensorflow as tf
            frameworks["TensorFlow"] = True
            frameworks["TensorFlow 版本"] = tf.__version__
            try:
                frameworks["TensorFlow GPU 支持"] = len(tf.config.list_physical_devices('GPU')) > 0
            except:
                frameworks["TensorFlow GPU 支持"] = False
                frameworks["Protobuf 版本问题"] = True
        except ImportError:
            pass
        except Exception as e:
            frameworks["TensorFlow"] = f"错误: {str(e)}"
            frameworks["Protobuf 版本问题"] = True

        # 检查 NumPy 兼容性
        try:
            import numpy as np
            frameworks["NumPy 版本"] = np.__version__
            if np.__version__.startswith('2.'):
                frameworks["NumPy 警告"] = "可能与其他库冲突"
        except ImportError:
            pass

        # 修改后的 JAX 检测
        if not CHECK_JAX:  # 如果配置为跳过JAX检测
            frameworks["JAX"] = "已跳过检测"
        else:
            try:
                import jaxlib
                import jax
                frameworks["JAX"] = True
                frameworks["JAX 版本"] = f"{jax.__version__} (jaxlib: {jaxlib.__version__})"
            except ImportError:
                pass
            except Exception as e:
                if "AVX" in str(e):
                    frameworks["JAX"] = "错误: CPU不支持AVX指令集"
                elif "_ARRAY_API" in str(e):
                    frameworks["JAX"] = "错误: NumPy版本不兼容"
                    frameworks["JAX 建议"] = "请执行: pip install 'numpy<2' 'jax[cpu]==0.4.13'"
                else:
                    frameworks["JAX"] = f"错误: {str(e)}"

            
        return frameworks

        # 检查PyTorch
        try:
            import torch
            frameworks["PyTorch"] = True
            frameworks["PyTorch 版本"] = torch.__version__
            frameworks["PyTorch GPU 支持"] = torch.cuda.is_available()
            if frameworks["PyTorch GPU 支持"]:
                frameworks["PyTorch GPU 数量"] = torch.cuda.device_count()
                frameworks["PyTorch 当前GPU"] = torch.cuda.current_device()
                frameworks["PyTorch GPU 名称"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
            
        # 检查Keras
        try:
            import keras
            frameworks["Keras"] = True
            frameworks["Keras 版本"] = keras.__version__
        except ImportError:
            pass
            
        # 检查OpenCV
        try:
            import cv2
            frameworks["OpenCV"] = True
            frameworks["OpenCV 版本"] = cv2.__version__
            frameworks["OpenCV CUDA 支持"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
        except ImportError:
            pass
            
        # 检查ONNX Runtime
        try:
            import onnxruntime
            frameworks["ONNX Runtime"] = True
            frameworks["ONNX Runtime 版本"] = onnxruntime.__version__
            providers = onnxruntime.get_available_providers()
            frameworks["ONNX Runtime 提供者"] = providers
            frameworks["ONNX Runtime GPU 支持"] = 'CUDAExecutionProvider' in providers
        except ImportError:
            pass
            
        # 检查CUDA
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                frameworks["CUDA"] = True
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'release' in line:
                        frameworks["CUDA 版本"] = line.split('release ')[1].split(',')[0]
        except FileNotFoundError:
            pass
            
        # 检查cuDNN
        try:
            import ctypes
            cudnn = ctypes.cdll.LoadLibrary('cudnn')
            frameworks["cuDNN"] = True
            # 获取cuDNN版本比较复杂，这里简化处理
        except Exception:
            pass
            
        return frameworks

    def check_face_recognition_libs(self):
        libs = {
            "face_recognition": False,
            "dlib": False,
            "OpenCV人脸检测": False,
            "MTCNN": False,
            "DeepFace": False,
            "InsightFace": False
        }
        
        # 检查face_recognition
        try:
            import face_recognition
            libs["face_recognition"] = True
            libs["face_recognition 版本"] = face_recognition.__version__
        except ImportError:
            pass
            
        # 检查dlib
        try:
            import dlib
            libs["dlib"] = True
            libs["dlib 版本"] = dlib.__version__
            libs["dlib GPU 支持"] = dlib.DLIB_USE_CUDA if hasattr(dlib, 'DLIB_USE_CUDA') else False
        except ImportError:
            pass
            
        # 检查OpenCV人脸检测
        try:
            import cv2
            libs["OpenCV人脸检测"] = True
        except ImportError:
            pass
            
        # 检查MTCNN
        try:
            from mtcnn import MTCNN
            libs["MTCNN"] = True
        except ImportError:
            pass
            
        # 检查DeepFace
        try:
            from deepface import DeepFace
            libs["DeepFace"] = True
            libs["DeepFace 版本"] = DeepFace.__version__
        except ImportError:
            pass
            
        # 检查InsightFace
        try:
            import insightface
            libs["InsightFace"] = True
            libs["InsightFace 版本"] = insightface.__version__
        except ImportError:
            pass
            
        return libs

    def evaluate_for_face_recognition(self):
        evaluation = {
            "硬件评估": {},
            "软件评估": {},
            "总体评分": 0,
            "建议": []
        }
        
        # 硬件评估
        cpu_cores = self.results['cpu_info']['核心数(物理)']
        memory_gb = self.results['memory_info']['总内存(GB)']
        has_gpu = len(self.results['gpu_info']) > 0 and not any('错误' in gpu for gpu in self.results['gpu_info'])
        
        # CPU评分
        cpu_score = min(cpu_cores / 4 * 25, 25)  # 4核得25分，每多1核加6.25分，最多25分
        
        # 内存评分
        memory_score = min(memory_gb / 8 * 25, 25)  # 8GB得25分，每多1GB加3.125分，最多25分
        
        # GPU评分
        gpu_score = 0
        if has_gpu:
            gpu_score = 25  # 有GPU得25分
            gpu_memory = float(self.results['gpu_info'][0]['显存(GB)'])
            gpu_score += min((gpu_memory - 4) / 4 * 25, 25)  # 4GB得0分，每多1GB加6.25分，最多25分
        
        evaluation['硬件评估']['CPU评分'] = cpu_score
        evaluation['硬件评估']['内存评分'] = memory_score
        evaluation['硬件评估']['GPU评分'] = gpu_score
        evaluation['硬件评估']['总分'] = cpu_score + memory_score + gpu_score
        
        # 软件评估
        frameworks = self.results['ai_frameworks']
        libs = self.results['face_recognition_libs']
        
        # 框架评分
        framework_score = 0
        if frameworks['TensorFlow'] or frameworks['PyTorch']:
            framework_score += 20
        if frameworks['OpenCV']:
            framework_score += 10
        if frameworks['CUDA'] and frameworks['cuDNN']:
            framework_score += 20
        
        # 库评分
        lib_score = 0
        if libs['face_recognition'] or libs['dlib']:
            lib_score += 15
        if libs['OpenCV人脸检测']:
            lib_score += 10
        if libs['MTCNN'] or libs['DeepFace'] or libs['InsightFace']:
            lib_score += 15
        
        evaluation['软件评估']['框架评分'] = framework_score
        evaluation['软件评估']['库评分'] = lib_score
        evaluation['软件评估']['总分'] = framework_score + lib_score
        
        # 总体评分
        total_score = evaluation['硬件评估']['总分'] * 0.6 + evaluation['软件评估']['总分'] * 0.4
        evaluation['总体评分'] = round(total_score, 1)
        
        # 生成建议
        if isinstance(frameworks.get('JAX'), str) and '初始化失败' in frameworks['JAX']:
            evaluation['建议'].append("检测到JAX库初始化问题，建议执行以下命令修复:")
            evaluation['建议'].append("1. pip uninstall jax jaxlib -y")
            evaluation['建议'].append("2. pip install --upgrade jax jaxlib")
        if isinstance(frameworks.get('JAX'), str) and '错误' in frameworks['JAX']:
            evaluation['建议'].append("检测到JAX库问题，建议执行以下命令修复:")
            evaluation['建议'].append("1. pip uninstall jax jaxlib -y")
            evaluation['建议'].append("2. pip install --upgrade jax jaxlib")
        if frameworks.get('JAX', False) and isinstance(frameworks['JAX'], str) and '错误' in frameworks['JAX']:
            evaluation['建议'].append("检测到JAX库初始化问题，可能需要重新安装: pip install --upgrade jax jaxlib")
        if cpu_cores < 4:
            evaluation['建议'].append("您的CPU核心数较少(少于4核)，可能影响人脸识别处理速度")
        if memory_gb < 8:
            evaluation['建议'].append("您的内存较小(小于8GB)，可能限制同时处理的人脸数量")
        if not has_gpu:
            evaluation['建议'].append("未检测到独立GPU，深度学习模型训练和推理速度会显著降低")
        elif float(self.results['gpu_info'][0]['显存(GB)']) < 4:
            evaluation['建议'].append("您的GPU显存较小(小于4GB)，可能限制模型大小和批量处理能力")
        
        if not (frameworks['TensorFlow'] or frameworks['PyTorch']):
            evaluation['建议'].append("未安装主流AI框架(TensorFlow/PyTorch)，建议安装其中之一")
        elif not (frameworks['CUDA'] and frameworks['cuDNN']):
            evaluation['建议'].append("检测到AI框架但未配置CUDA/cuDNN，无法利用GPU加速")
        
        if not (libs['face_recognition'] or libs['dlib'] or libs['OpenCV人脸检测']):
            evaluation['建议'].append("未安装主流人脸识别库(face_recognition/dlib/OpenCV)，建议至少安装一个")
        
        if total_score > 80:
            evaluation['建议'].insert(0, "您的设备非常适合人脸识别开发")
        elif total_score > 60:
            evaluation['建议'].insert(0, "您的设备适合人脸识别开发，但可能有性能限制")
        elif total_score > 40:
            evaluation['建议'].insert(0, "您的设备勉强可以用于人脸识别开发，但会有明显性能限制")
        else:
            evaluation['建议'].insert(0, "您的设备不太适合人脸识别开发，建议升级硬件或使用云服务")
        
        return evaluation


class PCConfigChecker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PC配置检测 - 人脸识别开发环境评估")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.init_ui()
        self.scanner = None
        
    def init_ui(self):
        # 标题
        title_label = QLabel("PC配置检测工具 - 人脸识别开发环境评估")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("准备就绪，点击开始扫描按钮检测系统配置")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)
        
        # 开始扫描按钮
        self.scan_button = QPushButton("开始扫描")
        self.scan_button.clicked.connect(self.start_scan)
        self.layout.addWidget(self.scan_button)
        
        # 保存结果按钮
        self.save_button = QPushButton("保存检测结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        self.layout.addWidget(self.save_button)
        
        # 选项卡
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # 创建选项卡
        self.summary_tab = QTextEdit()
        self.summary_tab.setReadOnly(True)
        self.tabs.addTab(self.summary_tab, "摘要")
        
        self.system_tab = QTextEdit()
        self.system_tab.setReadOnly(True)
        self.tabs.addTab(self.system_tab, "系统信息")
        
        self.hardware_tab = QTextEdit()
        self.hardware_tab.setReadOnly(True)
        self.tabs.addTab(self.hardware_tab, "硬件信息")
        
        self.software_tab = QTextEdit()
        self.software_tab.setReadOnly(True)
        self.tabs.addTab(self.software_tab, "软件信息")
        
        self.evaluation_tab = QTextEdit()
        self.evaluation_tab.setReadOnly(True)
        self.tabs.addTab(self.evaluation_tab, "评估与建议")
        
        # 底部信息
        footer_label = QLabel("© 2023 PC配置检测工具 - 专为人脸识别开发设计")
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setStyleSheet("color: gray;")
        self.layout.addWidget(footer_label)
        
    def start_scan(self):
        self.scan_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("扫描中...")
        
        # 清空之前的显示
        for i in range(self.tabs.count()):
            self.tabs.widget(i).clear()
        
        # 创建并启动扫描线程
        self.scanner = SystemScanner()
        self.scanner.update_signal.connect(self.update_progress)
        self.scanner.finished_signal.connect(self.scan_completed)
        self.scanner.start()
        
    def update_progress(self, message, progress):
        self.status_label.setText(message)
        self.progress_bar.setValue(progress)
        
    def scan_completed(self, results):
        self.results = results
        self.scan_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status_label.setText("扫描完成!")
        
        # 显示结果
        self.display_results(results)
        
    def display_results(self, results):
        # 摘要标签
        summary = f"系统扫描完成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        summary += f"操作系统: {results['system_info']['系统类型']} {results['system_info']['系统版本']}\n"
        summary += f"处理器: {results['cpu_info']['品牌']}\n"
        summary += f"物理核心: {results['cpu_info']['核心数(物理)']}, 逻辑核心: {results['cpu_info']['线程数(逻辑)']}\n"
        summary += f"内存: {results['memory_info']['总内存(GB)']} GB\n"
        
        if results['gpu_info'] and not any('错误' in gpu for gpu in results['gpu_info']):
            summary += f"GPU: {results['gpu_info'][0]['名称']} (显存: {results['gpu_info'][0]['显存(GB)']} GB)\n"
        else:
            summary += "GPU: 未检测到独立GPU或获取失败\n"
        
        score = results['evaluation']['总体评分']
        summary += f"\n人脸识别开发适用性评分: {score}/100\n"
        
        if score > 80:
            summary += "评价: 优秀 - 非常适合人脸识别开发"
        elif score > 60:
            summary += "评价: 良好 - 适合人脸识别开发"
        elif score > 40:
            summary += "评价: 一般 - 可以运行但可能有性能限制"
        else:
            summary += "评价: 较差 - 不太适合人脸识别开发"
        
        self.summary_tab.setPlainText(summary)
        
        # 系统信息标签
        system_info = "=== 系统基本信息 ===\n"
        for key, value in results['system_info'].items():
            system_info += f"{key}: {value}\n"
        
        self.system_tab.setPlainText(system_info)
        
        # 硬件信息标签
        hardware_info = "=== CPU信息 ===\n"
        for key, value in results['cpu_info'].items():
            hardware_info += f"{key}: {value}\n"
        
        hardware_info += "\n=== 内存信息 ===\n"
        for key, value in results['memory_info'].items():
            hardware_info += f"{key}: {value}\n"
        
        hardware_info += "\n=== 磁盘信息 ===\n"
        for disk in results['disk_info']:
            for key, value in disk.items():
                hardware_info += f"{key}: {value}\n"
            hardware_info += "\n"
        
        hardware_info += "\n=== GPU信息 ===\n"
        if results['gpu_info'] and not any('错误' in gpu for gpu in results['gpu_info']):
            for gpu in results['gpu_info']:
                for key, value in gpu.items():
                    hardware_info += f"{key}: {value}\n"
                hardware_info += "\n"
        else:
            hardware_info += "未检测到独立GPU或获取失败\n"
        
        self.hardware_tab.setPlainText(hardware_info)
        
        # 软件信息标签
        software_info = "=== AI框架支持 ===\n"
        for key, value in results['ai_frameworks'].items():
            software_info += f"{key}: {'是' if value is True else '否' if value is False else value}\n"
        
        software_info += "\n=== 人脸识别库支持 ===\n"
        for key, value in results['face_recognition_libs'].items():
            software_info += f"{key}: {'是' if value is True else '否' if value is False else value}\n"
        
        self.software_tab.setPlainText(software_info)
        
        # 评估与建议标签
        evaluation_info = "=== 硬件评估 ===\n"
        for key, value in results['evaluation']['硬件评估'].items():
            evaluation_info += f"{key}: {value}\n"
        
        evaluation_info += "\n=== 软件评估 ===\n"
        for key, value in results['evaluation']['软件评估'].items():
            evaluation_info += f"{key}: {value}\n"
        
        evaluation_info += f"\n=== 总体评分 ===\n{results['evaluation']['总体评分']}/100\n"
        
        evaluation_info += "\n=== 建议 ===\n"
        for suggestion in results['evaluation']['建议']:
            evaluation_info += f"- {suggestion}\n"
        
        self.evaluation_tab.setPlainText(evaluation_info)
        
    def save_results(self):
        if not hasattr(self, 'results'):
            QMessageBox.warning(self, "警告", "没有可保存的结果，请先执行扫描")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pc_config_check_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)
                
            QMessageBox.information(self, "成功", f"检测结果已保存到 {filename}")
        except Exception as e:
            print(f"保存文件时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存文件时出错: {str(e)}")


if __name__ == "__main__":
    # 检查protobuf版本
    try:
        import google.protobuf
        if google.protobuf.__version__ > "3.20.0":
            print("警告: 检测到高版本protobuf可能引起兼容性问题")
            print("建议执行: pip install protobuf==3.20.*")
    except ImportError:
        pass
        
    # 更安全的JAX检测
    try:
        import jax
        print(f"检测到JAX版本: {jax.__version__}")
    except ImportError:
        print("未检测到JAX库")
    except Exception as e:
        print(f"JAX初始化错误: {str(e)}")
        print("建议解决方案:")
        print("1. 完全卸载JAX: pip uninstall jax jaxlib -y")
        print("2. 重新安装: pip install --upgrade jax jaxlib")
    
    app = QApplication(sys.argv)

    
    # 检查必要库是否安装
    try:
        import cpuinfo
        import psutil
        import GPUtil
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        print(f"缺少必要依赖库: {str(e)}")
        print("请使用以下命令安装所需库:")
        print("pip install py-cpuinfo psutil gputil pyqt5")
        sys.exit(1)
    
    window = PCConfigChecker()
    window.show()
    sys.exit(app.exec_())