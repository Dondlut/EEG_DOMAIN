# 文件说明

## data

- patient_1.py  patient_2.py 和 patient_3.py 分别为训练集、验证集和测试集分配病人，由于随机分配过程中可能造成类别的严重失衡，所以手动调整了部分病人在三个子集中的分布。

- train_patient.txt  dev_patient.txt  和  test_patient.txt中分别展示了训练集、验证集和测试集中的病人。格式为  *病人id:{癫痫类别:癫痫样本数}*

## visualize

- ### fig

  - 可视化结果，文件夹名称代表癫痫类别

- ### h5_anno

  - 可视化样本，文件夹名称代表癫痫类别

- ### model

  - 训练过的模型

- lossFun.py 损失函数
- model  本文提出的模型
- temp  可视化代码

confusion_matrix.py用于生成混淆矩阵

constant_22.py用于获取患者id

train_c.py为训练过程