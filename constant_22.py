def getPatientFromPatient_txt(txt_path:str):
    '''
    获取患者id
    '''
    patient_id_list = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            patient_id = line.split(":")[0]
            patient_id_list.append(patient_id)
    return patient_id_list


train_id = getPatientFromPatient_txt('./data/train_patient.txt')
dev_id = getPatientFromPatient_txt('./data/dev_patient.txt')
test_id = getPatientFromPatient_txt('./data/test_patient.txt')