# Pose_Estimation
基于open-pose的轻量化版本的骨骼点生成器  
Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose  
代码基于Daniil-Osokin /lightweight-human-pose-estimation.pytorch  
  
(1) 环境配置  
torch>=0.4.1  
torchvision>=0.2.1  
pycocotools==2.0.0  
opencv-python>=3.4.0.14  
numpy>=1.14.0  
本项目采用了训练好的模型参数：  
参数下载链接：  
https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth  
若想要把.pth权重文件转化成onnx格式的权重文件，请在终端运行onnx.py文件  
python onnx.py .pth权重文件地址  
  
(2) 运行项目  
把测试视频文件放入video文件夹中，终端运行命令： python main.py 即可  
如若使用其他文件夹的视频文件，终端执行命令：python main.py --video 视频数据路径  
若想使用默认摄像头进行实时动作捕捉，终端执行命令：python main.py --video=0  
  
  
(3) 添加了计算“人体18个骨骼点”之间的向量角功能
