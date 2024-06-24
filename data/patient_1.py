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


def assignPatient(patient_dict: dict):
    samples_dict = {0: 24601, 1: 11020, 2: 140, 3: 1227}
    train_sample_dict = {0: 0, 1: 0, 2: 0, 3: 0}
    # # {患者id： {种类：数量, 种类：数量}, 患者id： {种类：数量, 种类：数量}}
    train_patient = {}
    for patient_id in patient_dict:
        for sz_class in patient_dict[patient_id].keys():
            if train_sample_dict[sz_class] < int(samples_dict[sz_class] * 0.7):
                train_sample_dict[sz_class] += patient_dict[patient_id][sz_class]

                if patient_id not in train_patient.keys():
                    train_patient[patient_id] = {}

                train_patient[patient_id][sz_class] = patient_dict[patient_id][sz_class]
    # print(train_patient)
    print(train_sample_dict)

    with open(r'train_patient.txt', 'w') as f:
        for key in train_patient.keys():
            f.write(key)  # patient_id
            f.write(":")
            f.write(str(train_patient[key]))
            f.write("\n")
    return train_patient


if __name__ == '__main__':
    # countSz()
    patient_dict = getAllPatientData()
    train_patient = assignPatient(patient_dict)
    for key in train_patient:
        if train_patient[key] != patient_dict[key]:
            print(key)
