import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

dataset = pd.read_csv('dataset.csv')
dataset = shuffle(dataset,random_state=42)

deskripsi = pd.read_csv('symptom_Description.csv')
precaution = pd.read_csv('symptom_precaution.csv')
severity = pd.read_csv('Symptom-severity.csv')
severity['Symptom'] = severity['Symptom'].str.replace('_',' ')
severity['Symptom'].unique()

for col in dataset.columns:
    
    dataset[col] = dataset[col].str.replace('_',' ')
    cols = dataset.columns
    data = dataset[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(dataset.shape)

    dataset = pd.DataFrame(s, columns=dataset.columns)
    dataset = dataset.fillna(0)

vals = dataset.values
symptoms = severity['Symptom'].unique()

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = severity[severity['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)
d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
dataset = d.replace('foul smell of urine',0)

data = dataset.iloc[:,1:].values
labels = dataset['Disease'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)

model = SVC(random_state=42)
model.fit(x_train, y_train)

preds = model.predict(x_test)

def predd(x, S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17):
    # Simpan daftar symptoms yang diinput user
    psymptoms = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17]
    
    # Hanya simpan symptoms yang valid (bukan 0)
    user_symptoms = [symptom for symptom in psymptoms if symptom != 0]  
    
    # Cek apakah jumlah gejala yang dimasukkan kurang dari 3
    if len(user_symptoms) < 3:
        print("Peringatan: Minimal 3 gejala harus dimasukkan!")
        return None, None, None  # Menghentikan eksekusi jika kurang dari 3 gejala
    
    # Konversi symptoms ke bobot
    a = np.array(severity["Symptom"])
    b = np.array(severity["weight"])
    for j in range(len(psymptoms)):
        for k in range(len(a)):
            if psymptoms[j] == a[k]:
                psymptoms[j] = b[k]

    # Membuat array untuk prediksi
    psy = [psymptoms]
    
    # Lakukan prediksi menggunakan model
    pred2 = x.predict(psy)
    
    # Ambil deskripsi penyakit
    disp = deskripsi[deskripsi['Disease'] == pred2[0]]
    disp = disp.values[0][1]
    
    # Ambil rekomendasi pencegahan
    recomnd = precaution[precaution['Disease'] == pred2[0]]
    c = np.where(precaution['Disease'] == pred2[0])[0][0]
    precaution_list = []
    for i in range(1, len(precaution.iloc[c])):
        precaution_list.append(precaution.iloc[c, i])
    
    # Output hasil
    print("Symptoms Input by User:")
    for symptom in user_symptoms:
        print(f"- {symptom}")
    
    print("\nThe Disease Name: ", pred2[0])
    print("The Disease Description: ", disp)
    print("Recommended Things to do at home: ")
    for i in precaution_list:
        print("- ", i)
    
    return pred2[0], disp, precaution_list

sympList=severity["Symptom"].to_list()
predd(model,sympList[122],sympList[19],sympList[22],0,0,0,0,0,0,0,0,0,0,0,0,0,0)
