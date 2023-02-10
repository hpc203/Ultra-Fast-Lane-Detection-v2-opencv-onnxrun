# Ultra-Fast-Lane-Detection-v2-opencv-onnxrun
分别使用OpenCV、ONNXRuntime部署Ultra-Fast-Lane-Detection-v2车道线检测，包含C++和Python两种版本的程序。

Ultra-Fast-Lane-Detection-v2是TPAMI2022期刊里的论文，它是速度精度双SOTA的最新车道线检测算法。

本套程序提供了分别在culane和tusimple数据集训练出来的模型，onnx模型文件在
百度云盘，链接：https://pan.baidu.com/s/1b68-q_NX2PQPHZQn2h-x2A 
提取码：jfwn


需要注意的是opencv版本需要是4.7以上的，opencv-dnn才能正常读取onnx文件做推理的
