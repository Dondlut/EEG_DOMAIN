import os

raw_data_dir = r''#h5_path



def getSamplesFromTxt(txt_path:str):
    ''''
    根据指定txt文件统计样本数
    '''
    patient_id_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            patient_id = line.split(":")[0]
            patient_id_list.append(patient_id)

    sz_dict = {}
    for fileName in os.listdir(raw_data_dir):
        sz_class = int(fileName.split("_")[-1].split(".")[0])  # 00000002_s001_t000.edf_0_1_0.h5
        patient_id = fileName.split("_")[0]
        if patient_id not in patient_id_list:
            continue
        if sz_class not in sz_dict.keys():
            sz_dict[sz_class] = 1
        else:
            sz_dict[sz_class] += 1

    print(txt_path)
    print(sz_dict)

getSamplesFromTxt(r'./train_patient.txt')
getSamplesFromTxt(r'./dev_patient.txt')
getSamplesFromTxt(r'./test_patient.txt')