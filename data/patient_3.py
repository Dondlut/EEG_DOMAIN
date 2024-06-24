import os
from random import shuffle

raw_data_dir = r''


def getAllPatientData():
    '''
    获取raw_data_dir下的所有患者的数据
    :return:
    '''
    # {患者id： {种类：数量, 种类：数量}, 患者id： {种类：数量, 种类：数量}}
    patient_dict = {}
    fileNum = 0
    for fileName in os.listdir(raw_data_dir):
        fileNum += 1
        sz_class = int(fileName.split("_")[-1].split(".")[0])  # 00000002_s001_t000.edf_0_1_0.h5
        patient_id = fileName.split("_")[0]
        if patient_id not in patient_dict.keys():
            patient_dict[patient_id] = {}
        if sz_class not in patient_dict[patient_id].keys():
            patient_dict[patient_id][sz_class] = 1
        else:
            patient_dict[patient_id][sz_class] += 1

    # print(patient_dict)
    # with open(r'./patient.txt', 'w') as f:
    #     for key in patient_dict.keys():
    #         f.write(key)  # patient_id
    #         f.write(":")
    #         f.write(str(patient_dict[key]))
    #         f.write("\n")
    return patient_dict


def countSz():
    '''
    统计raw_data_dir下的所有癫痫类别
    {0: 24601, 1: 11020, 2: 140, 3: 1227}

    :return:
    '''
    sz_dict = {}
    for fileName in os.listdir(raw_data_dir):
        sz_class = int(fileName.split("_")[-1].split(".")[0])  # 00000002_s001_t000.edf_0_1_0.h5
        if sz_class not in sz_dict.keys():
            sz_dict[sz_class] = 1
        else:
            sz_dict[sz_class] += 1
    print('癫痫样本数', sz_dict)


def assignPatient(patient_dict: dict, first_fold_patient_id_list : list, second_patient_id_list:list):
    '''
    为剩下两折分配病人
    :param patient_dict: 所有患者的数据
    :param dev_patient_id_list: 之前已经划分好的验证集的患者id列表
    :return:
    '''
    samples_dict = {0: 24601, 1: 11020, 2: 140, 3: 1227}
    first_fold_dict = {0: 9078, 1: 3642, 2: 54, 3: 371}  # 第一折样本，打算把这些样本排出后，再按患者分成两份，以此达到3折的目的
    second_fold_dict = {0: 7443, 1: 4044, 2: 43, 3: 468}  # 第二折样本
    third_fold_dict = {0:8080, 1:3334, 2:43, 3:388} # 预期

    # # {患者id： {种类：数量, 种类：数量}, 患者id： {种类：数量, 种类：数量}}
    third_fold_patient = {}
    for patient_id in patient_dict:
        if patient_id in first_fold_patient_id_list or patient_id in second_patient_id_list:  # 已经存在于之前的验证集，则跳过
            continue
        for sz_class in patient_dict[patient_id].keys():
                if patient_id not in third_fold_patient.keys():
                    third_fold_patient[patient_id] = {}

                third_fold_patient[patient_id][sz_class] = patient_dict[patient_id][sz_class]
    # print(second_fold_patient)
    # 验证总数是否正确
    for key in third_fold_patient.keys():
        for sz_class in third_fold_patient[key]:
            third_fold_dict[sz_class] -= third_fold_patient[key][sz_class]

    for key in third_fold_dict.keys():
        if third_fold_dict[key] != 0:
            print("error!")
            exit(-1)
    with open(r'test_patient.txt', 'w') as f:
        for key in third_fold_patient.keys():
            f.write(key)  # patient_id
            f.write(":")
            f.write(str(third_fold_patient[key]))
            f.write("\n")
    # print(third_fold_patient)
    return third_fold_patient


def checkPatientNumberBySz():
    '''
    根据癫痫类别计算患者数量
    :return:
    '''
    sz_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    # {患者id： {种类：数量, 种类：数量}, 患者id： {种类：数量, 种类：数量}}
    patient_dict = getAllPatientData()
    for patient_id in patient_dict.keys():
        for sz_class in patient_dict[patient_id].keys():
            sz_dict[sz_class] += 1
    print(sz_dict)


def getPatientFromFirst_fold_patient_txt():
    '''
    从first_fold_patient.txt中获取患者id，便于划分三折时进行排除
    :return:
    '''
    txt_path = r'train_patient.txt'
    first_patient_id_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            patient_id = line.split(":")[0]
            first_patient_id_list.append(patient_id)
    assert len(first_patient_id_list) == 20
    # print(first_patient_id_list)
    return first_patient_id_list

def getPatientFromSecond_fold_patient_txt():
    '''
    从first_fold_patient.txt中获取患者id，便于划分三折时进行排除
    :return:
    '''
    txt_path = r'dev_patient.txt'
    second_patient_id_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            patient_id = line.split(":")[0]
            second_patient_id_list.append(patient_id)
    assert len(second_patient_id_list) == 32
    # print(first_patient_id_list)
    return second_patient_id_list

if __name__ == '__main__':
    # countSz()
    # checkPatientNumberBySz()
    # getPatientFromFirst_fold_patient_txt()
    patient_dict = getAllPatientData()
    first_patient_id_list = getPatientFromFirst_fold_patient_txt()
    second_patient_id_list = getPatientFromSecond_fold_patient_txt()
    third_fold_patient = assignPatient(patient_dict,first_patient_id_list, second_patient_id_list)

