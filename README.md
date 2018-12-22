# mtcnn_numpy_implement

## 前言
    
    主要是为了学习MTCNN的三个网络模型组成，以及各层的输入和输出的维度，并使用numpy来模拟各层的运算过程。
    
## 使用

在Net_blob_verify.py文件中compare_PNet_blob_with_numpy、compare_QNet_blob_with_numpy、compare_ONet_blob_with_numpy函数分别实现了

PNet、RNet、ONet三个网络的各层。运行时需要先去掉对应函数的注释，再运行该文件即可。

** 注：compare_PNet_blob_with_numpy、compare_QNet_blob_with_numpy、compare_ONet_blob_with_numpy函数运行时间较长 **

## 鸣谢

* https://github.com/ahmedfgad/NumPyCNN 主要依赖这个链接来实现卷积、池化等函数（一些函数的运行结果跟Caffe有出入，因此本地跟repo不同）
* https://github.com/DuinoDu/mtcnn 代码的主体框架，基本按这个来的。

## 补充

1. 后续会增加演示采用了compare_PNet_blob_with_numpy、compare_QNet_blob_with_numpy、compare_ONet_blob_with_numpy三个函数的检测效果图
2. 分析下这个https://github.com/espressif/esp-who 链接的在esp-32芯片的MTCNN实现。
3. https://github.com/zjd1988/mtcnn_vs2017_based_on_ncnn 这是我在windows端编译通过的vs版本，可以单步调试，后续会学习下在ARM上的实现。



