# mtcnn_numpy_implement

## 前言

主要是为了学习MTCNN的三个网络模型组成，以及各层的输入和输出的维度，并使用numpy来模拟各层的运算过程。
    
## 使用

在Net_blob_verify.py文件中compare_PNet_blob_with_numpy、compare_QNet_blob_with_numpy、compare_ONet_blob_with_numpy函数分别实现了PNet、RNet、ONet三个网络的各层。运行时需要先去掉对应函数的注释，再运行该文件即可。

**注：compare_PNet_blob_with_numpy、compare_QNet_blob_with_numpy、compare_ONet_blob_with_numpy函数运行时间较长**

## 运行结果

Elapsed time: 1133.5085640069228 seconds

![result](https://github.com/zjd1988/mtcnn_numpy_implement/blob/master/result.jpg)

## 鸣谢

* https://github.com/pierluigiferrari/caffe_weight_converter 使用这个工具将Caffemodel文件的权重，保存为pkl格式文件，方便numpy导入和读取
* https://github.com/ahmedfgad/NumPyCNN 主要依赖这个链接来实现卷积、池化等函数（一些函数的运行结果跟Caffe有出入，因此本地跟repo不同）
* https://github.com/DuinoDu/mtcnn 代码的主体框架，基本按这个来的。

## 补充

1. 分析下这个https://github.com/espressif/esp-who 链接的在esp-32芯片的MTCNN实现。
2. https://github.com/zjd1988/mtcnn_vs2017_based_on_ncnn 这是我在windows端编译通过的vs版本，可以单步调试，后续会学习下在ARM上的实现。
3. git rebase