* 来自计算机视觉老师的作业，使用网络对CIFAR10数据集进行识别查看准确率
使用了五个模型
1. 基于Channel_Spatial_Attention的简单ResNet网络
2. ResNet50预训练模型只修改最后的全连接层
3. ResNet50全连接层前添加具有Channel_Spatial_Attention的ResNet模块
4. ResNet50从头训练
5. 添加注意力机制的ResNet50从头训练

流程
1. 运行data.py可以将数据下载到文件名为cifar目录下
2. 使用数据增强进行训练防止过拟合
3. 运行train.py给定模型参数可以使用不同模型进行训练
4. 训练过后的日志和训练好的模型参数分别保存在Log和train_model下
5. 运行classify.py可以对单个图片进行分类

精度<br>
Sample ResNet  Train Acc : 0.7605  Val Acc : 0.6609<br>
ResNet50       Train Acc : 0.8647  Val Acc : 0.8443<br>
ResNet Attention Train Acc : 0.9457  Val Acc : 0.8854<br>
ResNet50 scratch Train Acc : 0.9914  Val Acc : 0.9613<br>
ResNet Attention scratch Train Acc : 0.9952  Val Acc : 0.9631<br>
