import os
from dataloader_22_1 import test_loader
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from confusion_matrix import plot_confusion_matrix
from model import DANNModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
torch.cuda.set_device(3)
device = torch.device("cuda:3")
print('current_device: ', torch.cuda.current_device())
print('device_count(): ', torch.cuda.device_count())
LongTensor = torch.cuda.LongTensor
FloatTensor = torch.cuda.FloatTensor
i = 1
with open(r'./test_samples.txt', 'w') as f:

    while i <= 1:
        model = DANNModel(nb_classes=4, dropOut=None)
        model_name = "model_{}".format(16)
        model = model.cuda()
        model.load_state_dict(torch.load(r'./{}.pth'.format(model_name), map_location='cuda:3'), False)
        model.eval()
        patient_id_list = []
        with torch.no_grad():
            test_loss = 0
            test_total_loss = 0
            acc = 0
            label_sum = 0  # 记录验证集的数据量
            test_sub_loss = 0
            sub_acc = 0

            for batch_idx, (x, y, patient_id,sz_file) in enumerate(test_loader):
                alpha = 0
                img = x
                # print(sz_file)
                # print(type(sz_file))

                img = img.type(FloatTensor).cuda()
                img = img.permute(0, 3, 1, 2)
                img = torch.unsqueeze(img, 1)

                label = y.cuda().type(LongTensor)

                dev_data_nums = label.shape[0]
                label_sum += dev_data_nums
                # print((type(patient_id)))
                # print(patient_id)
                patient_id_list = patient_id_list + list(patient_id)
                classOutPut, _ = model(img, alpha)
                pred = torch.max(classOutPut, 1)[1]
                acc += float((pred == label).cpu().numpy().astype(int).sum())

                for j in range(dev_data_nums):
                    if label[j] == pred[j]:
                        f.write(str(sz_file[j]) + " " + str(label[j]))
                        f.write("\n")

                if batch_idx == 0:
                    y_pre = pred
                    y_true = label
                else:
                    y_pre = torch.cat((y_pre, pred))
                    y_true = torch.cat((y_true, label))

            acc = acc / float(label_sum)

            print('test accuracy is:', acc)

        # plot_confusion_matrix(y_true, y_pre, "{}".format(model_name))
        F1_score_metrics = f1_score(y_true.cpu().detach().numpy(),
                                    y_pre.cpu().detach().numpy(),
                                    average="weighted")

        # 计算相关指标
        precision = precision_score(y_true.cpu().detach().numpy(),
                                    y_pre.cpu().detach().numpy(),
                                    average="weighted")

        recall = recall_score(y_true.cpu().detach().numpy(),
                              y_pre.cpu().detach().numpy(),
                              average="weighted")
        print("test weighted-F1 is: ", F1_score_metrics)
        print("test precision is: ", precision)
        print("test recall is: ", recall)
        i += 1
        print("------up i={}--------".format(i-1))



