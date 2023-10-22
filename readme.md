* 来自计算机视觉老师的作业，使用网络对CIFAR10数据集进行识别查看准确率
使用了三个模型
1. 基于Channel_Spatial_Attention的简单ResNet网络
2. ResNet50预训练模型只修改最后的全连接层
3. ResNet50全连接层前添加具有Channel_Spatial_Attention的ResNet模块

流程
1. 运行data.py可以将数据下载到文件名为cifar目录下
2. 使用数据增强进行训练防止过拟合
3. 运行train.py给定模型参数可以使用不同模型进行训练
4. 训练过后的日志和训练好的模型参数分别保存在Log和train_model下
5. 运行classify.py可以对单个图片进行分类

精度
1. Sample ResNet  Train Acc : 0.7605  Val Acc : 0.6609
2. ResNet50       Train Acc : 0.8647  Val Acc : 0.8443
3. ResNet Attention Train Acc : 0.9457  Val Acc : 0.8854
