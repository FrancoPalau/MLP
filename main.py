import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def f_no_lineal(a,b):
    return a*np.sin(b)

def f(x):
    return 1 / (1 + np.exp(-x))

def f_derivada(x):
     return np.exp(-x)/((1+np.exp(-x))**2)

def g(x):
    return f(x)

def g_derivada(x):
    return f_derivada(x)

def calculo_salida(Wji, Wkj, serie_ejemplo):

    output_capa_media = []
    for j in range(NEURONAS_CAPA_OCULTA):
        regla_propagacion_y = 0
        for i in range(NEURONAS_ENTRADA):
            regla_propagacion_y += Wji[i][j] * serie_ejemplo[i]
        regla_propagacion_y -= Wji[NEURONAS_ENTRADA][j]  # Restamos el sesgo
        output_capa_media.append(f(regla_propagacion_y))

    output_capa_salida = []
    for k in range(NEURONAS_SALIDA):
        regla_propagacion_z = 0
        for j in range(NEURONAS_CAPA_OCULTA):
            regla_propagacion_z += Wkj[j][k] * output_capa_media[j]
        regla_propagacion_z -= Wkj[NEURONAS_CAPA_OCULTA][k]  # Restamos el sesgo
        output_capa_salida.append(g(regla_propagacion_z))

    return output_capa_media, output_capa_salida

def backpropagation(Wji, Wkj, serie_ejemplo, valor_deseado, output_capa_media):

    delta_mu_k = []
    for k in range(NEURONAS_SALIDA):
        h_mu_k = 0
        for j in range(NEURONAS_CAPA_OCULTA):
            h_mu_k += Wkj[j][k] * output_capa_media[j]
        h_mu_k -= Wkj[NEURONAS_CAPA_OCULTA][k]
        aux = (valor_deseado - g(h_mu_k)) * g_derivada(h_mu_k)
        delta_mu_k.append(aux)
        for j in range(NEURONAS_CAPA_OCULTA):
            Wkj[j][k] += EPSILON * delta_mu_k[k] * output_capa_media[j]
        Wkj[NEURONAS_CAPA_OCULTA][k] += EPSILON * delta_mu_k[k] * -1

    for j in range(NEURONAS_CAPA_OCULTA):
        h_mu_j = 0
        for i in range(NEURONAS_ENTRADA):
            h_mu_j += Wji[i][j] * serie_ejemplo[i]
        h_mu_j -= Wji[NEURONAS_ENTRADA][j]
        delta_mu_j = 0
        for k in range(NEURONAS_SALIDA):
            delta_mu_j += delta_mu_k[k] * Wkj[j][k]
        delta_mu_j *= f_derivada(h_mu_j)
        for i in range(NEURONAS_ENTRADA):
            Wji[i][j] += EPSILON * delta_mu_j * serie_ejemplo[i]
        Wji[NEURONAS_ENTRADA][j] += EPSILON * delta_mu_j * -1

def main():
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    np.random.seed(10)
    #Cargamos dataset
    #data = pd.read_excel("EjemplosEntrenamiento10.xlsx")
    data = pd.read_csv("Ecommerce Customers")
    # X = data[data.columns[3:7]]
    # y = data[data.columns[7]]
    data.drop(["Email", "Address", "Avatar"], axis=1, inplace=True)

    # # Creacion del dataset
    # data = pd.DataFrame(np.random.randint(low=-100, high=200, size=(2000, 2)), columns="C1 C2".split())
    # data["Y"] = np.vectorize(f_no_lineal)(data["C1"], data["C2"])

    # Escalamos las entradas (media cero y desviacion estandar 1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data)
    data_feat = pd.DataFrame(scaled_features, columns=data.columns)

    # #Normalizamos entre [0,1]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data_feat)
    data_feat = pd.DataFrame(scaled_features, columns=data_feat.columns)

    # Inicializamos los pesos
    Wji = np.random.randint(low=-100, high=100, size=(NEURONAS_ENTRADA + 1, NEURONAS_CAPA_OCULTA))/10000   # +1 por el sesgo
    Wkj = np.random.randint(low=-100, high=100, size=(NEURONAS_CAPA_OCULTA + 1, NEURONAS_SALIDA))/10000

    # Dividimos el dataset en entrenamiento y testeo(X=entradas, y=salidas)
    X_train, X_test, y_train, y_test = train_test_split(data_feat[data_feat.columns[:-1]], data_feat["Yearly Amount Spent"], test_size=0.3, random_state=101)

    epoc = []
    ratio_mean = []
    for i in range(1000):
        # ----- Entrenamiento ------
        for ejemplo in X_train.index:
            serie_ejemplo = X_train.loc[ejemplo]
            valor_deseado = y_train.loc[ejemplo]
            output_capa_media,output_capa_salida = calculo_salida(Wji,Wkj,serie_ejemplo)
            backpropagation(Wji,Wkj,serie_ejemplo,valor_deseado,output_capa_media)

        # ----- Testeo ------
        ratio = []
        predictions = []
        for num,indice in enumerate(X_test.index):
            prueba = X_test.loc[indice]
            output_capa_media, output_capa_salida = calculo_salida(Wji, Wkj,prueba)
            print(output_capa_salida, y_test.loc[indice])
            predictions.append(output_capa_salida)
            if y_test.loc[indice] != 0:
                ratio.append((abs(y_test.loc[indice] - predictions[num]))/abs(y_test.loc[indice]))
        print("RATIO:", np.mean(ratio))
        epoc.append(i)
        ratio_mean.append(np.mean(ratio))
    plt.plot(epoc,ratio_mean)
    plt.show()

if __name__ == "__main__" :

    NEURONAS_ENTRADA = 4
    NEURONAS_CAPA_OCULTA = 5
    NEURONAS_SALIDA = 1
    EPSILON = 0.4

    main()

