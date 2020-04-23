import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.model_selection import train_test_split

from scipy.stats import chi
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df = pd.read_csv(url, names=['target', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_Ash',
                             'Magnesium', 'Total_Phenots', 'Flavanoids',
                             'Nonflavanoid_phenols', 'Proanthocyanins',
                             'Color_Intensity', 'Hue', 'OD280_OD315',
                             'Proline'])
target = df.target
df = df.drop(['target'], axis=1)
df = ((df-df.min()) / (df.max()-df.min()))
X_train, X_test, y_train, y_test = train_test_split(df, target, train_size=0.7, random_state=True)


def bin2float(str_bin, dec_sign):
    # This function turn a binary string into a float with dec_sign integer digits
    bin = 0
    if (str_bin[0] == '1'):

        sig = 1
    else:
        sig = -1
    for i in range(1, len(str_bin)):
        tmp = int(str_bin[i:i + 1])
        bin = bin * 2 + tmp
    max_num_repre = int(np.power(2, len(str_bin) - 1)) - 1
    str_max_num_repre = str(max_num_repre)
    max = int(np.power(10, len(str_max_num_repre) - dec_sign))
    return sig * (bin / max)

def eucli_dins(p1, p2):
    return np.sqrt((np.power(p1[0] - p2[0], 2)) + (np.power(p1[1] - p2[1], 2)))


def fitness(weights):
    layer = np.ndarray(shape=(1, 3), dtype= np.float)
    dec_vec = []
    vec_out = []
    ran_sta = np.random.randint(0, 10000)
    X_train_tmp = pd.DataFrame.sample(X_train, n=X_train.shape[0], random_state=ran_sta)
    y_train_tmp = list(pd.DataFrame.sample(y_train, n=y_train.shape[0], random_state=ran_sta))
    for ind in weights:
        dec_vec.append(bin2float(ind, 1))
    for row in range(X_train.shape[0]):
        cont_w = 0
        x0 = dec_vec[cont_w]
        cont_w += 1
        x1 = dec_vec[cont_w]
        cont_w += 1
        x2 = dec_vec[cont_w]
        cont_w += 1
        for col in range(X_train.shape[1]):
            x0 += dec_vec[cont_w] * X_train_tmp.iloc[row, col]
            cont_w += 1
            x1 += dec_vec[cont_w] * X_train_tmp.iloc[row, col]
            cont_w += 1
            x2 += dec_vec[cont_w] * X_train_tmp.iloc[row, col]
            cont_w += 1
        x0 = expit(x0)
        x1 = expit(x1)
        x2 = expit(x2)
        out = list(softmax([x0, x1, x2]))
        vec_out.append(out.index(max(out))+1)
    acc = 0
    for sol in range(len(y_train)):
        if y_train_tmp[sol] == vec_out[sol]:
            acc += 1
    return float(acc)/float(len(y_train))


def evaluation_test(weights):
    dec_vec = []
    vec_out = []
    y_test_tmp = list(y_test)
    for i in weights:
        dec_vec.append(bin2float(i, 1))
    for row in range(X_test.shape[0]):
        cont_w = 0
        x0 = dec_vec[cont_w]
        cont_w += 1
        x1 = dec_vec[cont_w]
        cont_w += 1
        x2 = dec_vec[cont_w]
        cont_w += 1
        for col in range(X_test.shape[1]):
            x0 += dec_vec[cont_w] * X_test.iloc[row, col]
            cont_w += 1
            x1 += dec_vec[cont_w] * X_test.iloc[row, col]
            cont_w += 1
            x2 += dec_vec[cont_w] * X_test.iloc[row, col]
            cont_w += 1
        x0 = expit(x0)
        x1 = expit(x1)
        x2 = expit(x2)
        out = list(softmax([x0, x1, x2]))
        vec_out.append(out.index(max(out))+1)
    acc = 0
    for sol in range(len(y_test)):
        if y_test_tmp[sol] == vec_out[sol]:
            acc += 1
    return float(acc)/float(len(y_test))


