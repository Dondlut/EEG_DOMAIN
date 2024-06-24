import os
import numpy as np
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score
from dataloader_22_1 import source_loader, target_loader
from model import DANNModel

# from sklearn.metrics import accuracy_score
# from lossFun import ContrastiveLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
torch.cuda.set_device(0)
device = torch.device("cuda:0")
print('current_device: ', torch.cuda.current_device())
print('device_count(): ', torch.cuda.device_count())
seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tb_writer = SummaryWriter()
model_count = 1
n_epochs = 100
lr = 0.0001

# source domain： {0: 16513, 1: 7753, 2: 97, 3: 956}
# 401
# target domain： {0: 5609, 1: 2160, 2: 27, 3: 180}
# 398
# test domain： {0: 2479, 1: 1107, 2: 16, 3: 91}


weight_class = [1.0, 2.13, 170.24, 17.27]
loss_class = nn.NLLLoss(weight=torch.tensor(weight_class)).cuda()
weight_domain = [1.0, 3.17]
loss_domain = nn.NLLLoss(weight=torch.tensor(weight_domain)).cuda()

loss_test = nn.NLLLoss().cuda()

LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor
model = DANNModel(nb_classes=4, dropOut=None)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs)

bestAcc = 0
averAcc = 0
num = 0
bestWeightedF1 = 0

for epoch in range(n_epochs):
    print('-----------no.{} epoch\'s train starts!---------'.format(epoch + 1))

    in_epoch = time.time()
    # 训练集准确率
    train_acc = 0
    # subject 准确率
    sub_acc = 0
    # 总loss
    total_loss = 0
    # 训练集分类loss
    err_label_total = 0
    # domain_loss
    err_domain_total = 0
    # 训练集数据总量
    train_data_nums = 0

    len_dataloader = min(len(source_loader), len(target_loader))

    model.train()

    in_time = time.time()
    for batch_idx, ((x_source, label_source, domain_source), (x_target, _, domain_target)) in enumerate(
            zip(source_loader, target_loader)):
        optimizer.zero_grad()
        p = float(batch_idx + epoch * len_dataloader) / n_epochs / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # source
        train_data_nums += x_source.shape[0]
        img_source = x_source
        # img_source = torch.unsqueeze(img_source,1)
        # print('img_source.shape',img_source.shape)
        img_source = img_source.type(FloatTensor).cuda()
        img_source = img_source.permute(0, 3, 1, 2)
        img_source = torch.unsqueeze(img_source, dim=1)

        label_source = label_source.cuda().type(LongTensor)
        domain_source = domain_source.cuda().type(LongTensor)

        classOutPut, domainOutPut = model(img_source, alpha)
        pred = torch.max(classOutPut, 1)[1]

        train_acc += float((pred == label_source).cpu().numpy().astype(int).sum())

        err_s_label = loss_class(classOutPut, label_source)
        err_s_domain = loss_domain(domainOutPut, domain_source)

        # target
        img_target = x_target
        # img_target = torch.unsqueeze(img_target,1)
        img_target = img_target.type(FloatTensor).cuda()
        img_target = img_target.permute(0, 3, 1, 2)
        img_target = torch.unsqueeze(img_target, dim=1)

        domain_target = domain_target.cuda().type(LongTensor)
        _, domainOutPut = model(img_target, alpha)
        err_t_domain = loss_domain(domainOutPut, domain_target)

        err_label_total += err_s_label  # tensorboard
        err_domain_total += (err_s_domain + err_t_domain)  # tensorboard

        err = err_t_domain + err_s_domain + err_s_label
        total_loss += err.item()
        err.backward()
        optimizer.step()

    scheduler.step()
    out_epoch = time.time()
    print('第{}个epoch训练时间为{}秒'.format(epoch + 1, out_epoch - in_epoch))
    train_acc /= train_data_nums
    tb_writer.add_scalar("label_loss", err_label_total, epoch + 1)
    tb_writer.add_scalar("domain_loss", err_domain_total, epoch + 1)
    tb_writer.add_scalar("total_loss", total_loss, epoch + 1)
    tb_writer.add_scalar("train_acc", train_acc, epoch + 1)
    print('Epoch:', epoch + 1, 'Total loss: ', total_loss, 'Train acc: ', train_acc)

    # training model using target data
    print('-----------no.{} epoch\'s eva starts!---------'.format(epoch + 1))
    model.eval()
    with torch.no_grad():
        test_loss = 0
        test_total_loss = 0
        acc = 0
        label_sum = 0  # 记录验证集的数据量
        test_sub_loss = 0

        for batch_idx, (x, y_label, y_domain) in enumerate(target_loader):
            alpha = 0

            img = x

            img = img.type(FloatTensor).cuda()
            img = img.permute(0, 3, 1, 2)
            img = torch.unsqueeze(img, dim=1)

            # img = torch.unsqueeze(img, 1)
            label = y_label.cuda().type(LongTensor)

            label_sum += label.shape[0]

            classOutPut, _ = model(img, alpha)
            pred = torch.max(classOutPut, 1)[1]
            acc += float((pred == label).cpu().numpy().astype(int).sum())

            lossFirst = loss_test(classOutPut, label)
            test_total_loss = lossFirst
            test_loss += lossFirst.item()

            if batch_idx == 0:
                y_pre = pred
                y_true = label
            else:
                y_pre = torch.cat((y_pre, pred))
                y_true = torch.cat((y_true, label))

        test_loss /= label_sum
        acc = acc / float(label_sum)

        tb_writer.add_scalar("test_loss", test_loss, epoch + 1)
        tb_writer.add_scalar("acc", acc, epoch + 1)

        num = num + 1
        averAcc = averAcc + acc
        if acc > bestAcc:
            bestAcc = acc
            # torch.save(model.state_dict(), 'best_model.pth')

        F1_score_metrics = f1_score(y_true.cpu().detach().numpy(),
                                    y_pre.cpu().detach().numpy(),
                                    average="weighted")

        print('Epoch:', epoch + 1,
              '  Test loss:', test_loss,
              '  acc is:', acc,
              '  F1_score_metrics is :', F1_score_metrics)
        if F1_score_metrics > bestWeightedF1:
            bestWeightedF1 = F1_score_metrics
            torch.save(model.state_dict(), 'model.pth')
        tb_writer.add_scalar("dev_weighted_F1-score", F1_score_metrics, epoch + 1)

averAcc = averAcc / num
print('The average accuracy is:', averAcc)
print('The best accuracy is:', bestAcc)
