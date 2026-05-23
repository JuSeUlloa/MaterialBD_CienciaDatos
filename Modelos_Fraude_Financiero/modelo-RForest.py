import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from datetime import datetime
import scipy
scipy.interp = np.interp
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, make_scorer
from sklearn.metrics import recall_score, precision_score, log_loss, confusion_matrix, classification_report, roc_auc_score, accuracy_score
from plot_metric.functions import BinaryClassification
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
import xgboost
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


path = "data/"


base = pd.read_csv(path + "train.csv",header=0,sep=";",decimal=",")
prueba = pd.read_csv(path + "test.csv",header=0,sep=";",decimal=",")

ft = pd.DataFrame(base.isnull().sum()).reset_index()
ft.columns = ["Variable","Faltantes"]
ft["% Faltantes"] = ft["Faltantes"] * 100 / base.shape[0]

formato = pd.DataFrame({'Variable': list(base.columns), 'Formato': base.dtypes })
ft = pd.merge(ft,formato,on=["Variable"],how="left")
ft.loc[ft["% Faltantes"]>0,]


ft1 = pd.DataFrame(prueba.isnull().sum()).reset_index()
ft1.columns = ["Variable","Faltantes"]
ft1["% Faltantes"] = ft1["Faltantes"] * 100 / prueba.shape[0]


formato2 = pd.DataFrame({'Variable': list(prueba.columns), 'Formato': prueba.dtypes })
ft1 = pd.merge(ft1,formato2,on=["Variable"],how="left")
ft1.loc[ft1["% Faltantes"]>0,]

inputacion = ft.loc[ft["% Faltantes"]>0,].copy()
inputacion.index = range(inputacion.shape[0])

for k in range(inputacion.shape[0]):
    if inputacion.iloc[k,3] == "str":
        calculo = base.loc[base[inputacion.iloc[k,0]].isnull()==False,inputacion.iloc[k,0]]
        moda = calculo.describe()['top']
        base.loc[base[inputacion.iloc[k,0]].isnull()==True,inputacion.iloc[k,0]] = moda
        prueba.loc[prueba[inputacion.iloc[k,0]].isnull()==True,inputacion.iloc[k,0]] = moda
    else:
        calculo2 = base.loc[base[inputacion.iloc[k,0]].isnull()==False,inputacion.iloc[k,0]]
        mediana_ = np.median(calculo2)
        base.loc[base[inputacion.iloc[k,0]].isnull()==True,inputacion.iloc[k,0]] = mediana_
        prueba.loc[prueba[inputacion.iloc[k,0]].isnull()==True,inputacion.iloc[k,0]] = mediana_

ft = pd.DataFrame(base.isnull().sum()).reset_index()
ft.columns = ["Variable","Faltantes"]
ft["% Faltantes"] = ft["Faltantes"] * 100 / base.shape[0]
ft.loc[ft["% Faltantes"]>0,]

ft = pd.DataFrame(prueba.isnull().sum()).reset_index()
ft.columns = ["Variable","Faltantes"]
ft["% Faltantes"] = ft["Faltantes"] * 100 / prueba.shape[0]
ft.loc[ft["% Faltantes"]>0,]


cuantitativas = list(formato.loc[formato["Formato"]!="string","Variable"])
cuantitativas = [x for x in cuantitativas if x not in ["ID","Fraude"]]

#print("Variables cuantitativas:", cuantitativas)

categoricas = list(formato.loc[formato["Formato"]=="str","Variable"])
#print("Variables cualitativas:",categoricas)


from scipy import stats
def asociacion(data):
    categoricas=data.columns[data.dtypes=='str']
    V1=np.array(categoricas); V2=np.array(categoricas)
    grilla=np.meshgrid(V1,V2)
    grilla=pd.DataFrame({'Var1':grilla[0].ravel(),'Var2':grilla[1].ravel()})
    p_value=[stats.chi2_contingency(pd.DataFrame(pd.crosstab(data[grilla.iloc[x]['Var1']],data[grilla.iloc[x]['Var2']])))[1] for x in range(grilla.shape[0])]
    
    grilla['p_value']=p_value
    grilla2=grilla.pivot(index='Var1', columns='Var2', values='p_value')
    plt.figure(figsize=(10,8))
    gr=sns.heatmap(grilla2,linewidths=0.01,annot=True,fmt='.2f',cmap='summer')
    gr.set_title('Grilla de p valores en prueba chi cuadrado para verificar asociaciÃ³n entre variables')
    plt.xlabel(""); plt.ylabel(""); plt.yticks(rotation=0); plt.xticks(rotation=90)
    plt.show()
    return grilla,grilla2


#print(base.get(cuantitativas).describe())

## Prueba de Kruskal para variables cuantitativas base cuadrado
from scipy import stats
from itertools import combinations

base_cuadrado = base.get(cuantitativas).copy()
base_cuadrado["Fraude"] = base["Fraude"].copy()

var_names2, pvalue1 = [], []
variables_escaladas =[]

for k in cuantitativas:

    max_val =np.max(base_cuadrado[k])

    if max_val >1000:
        base_cuadrado[k+"_2"] = (base_cuadrado[k] /1000) ** 2
        variables_escaladas.append(k)
    else:
        base_cuadrado[k+"_2"] = base_cuadrado[k] ** 2
        
    # Prueba de Kruskal sin logaritmo
    mue1 = base_cuadrado.loc[base_cuadrado["Fraude"]==1,k+"_2"].to_numpy()
    mue2 = base_cuadrado.loc[base_cuadrado["Fraude"]==0,k+"_2"].to_numpy()
    p1 = stats.kruskal(mue1,mue2)[1]
    
    if p1 < 0.05:
        var_names2.append(k+"_2")
        pvalue1.append(np.round(p1,2))


pcuadrado1 = pd.DataFrame({'Variable2':var_names2,'p value':pvalue1})

### Interacciones
from itertools import combinations
lista_inter = list(combinations(cuantitativas,2))

base_interacciones = base.get(cuantitativas).copy()
base_interacciones["Fraude"] = base["Fraude"].copy()

variables_inter_escaladas = []

for j in cuantitativas:
    
    if np.max(base_interacciones[j]>1000):
        base_interacciones[j] = base_interacciones[j] / 1000
        variables_inter_escaladas.append(j)
    
var_interaccion, pv1 = [], []
for k in lista_inter:
    base_interacciones[k[0]+"_"+k[1]] = base_interacciones[k[0]] * base_interacciones[k[1]]
    
    # Prueba de Kruskal
    mue1 = base_interacciones.loc[base_interacciones["Fraude"]==1,k[0]+"_"+k[1]].to_numpy()
    mue2 = base_interacciones.loc[base_interacciones["Fraude"]==0,k[0]+"_"+k[1]].to_numpy()
    p1 = stats.kruskal(mue1,mue2)[1]
    
    if p1 < 0.05:
        var_interaccion.append(k[0]+"_"+k[1])
        pv1.append(np.round(p1,2))

pxy = pd.DataFrame({'Variable':var_interaccion,'p value':pv1})

### Razones
raz1 = [(x,y) for x in cuantitativas for y in cuantitativas]

base_razones1 = base.get(cuantitativas).copy()
base_razones1["Fraude"] = base["Fraude"].copy()

for col in cuantitativas:
    if np.max(base_razones1[col]) > 1000:
        base_razones1[col] = base_razones1[col] / 1000

var_nm, pval = [], []

for j in raz1:
    if j[0]!=j[1]:
        base_razones1[j[0]+"_coc_"+j[1]] = base_razones1[j[0]] / (base_razones1[j[1]]+0.01) 
        
        # Prueba de Kruskal
        mue1 = base_razones1.loc[base_razones1["Fraude"]==1,j[0]+"_coc_"+j[1]].to_numpy()
        mue2 = base_razones1.loc[base_razones1["Fraude"]==0,j[0]+"_coc_"+j[1]].to_numpy()
        p1 = stats.kruskal(mue1,mue2)[1]
        
        # Guardar valores
        if p1 < 0.05:
            var_nm.append(j[0]+"_coc_"+j[1])
            pval.append(np.round(p1,2))
        

prazones = pd.DataFrame({'Variable':var_nm,'p value':pval})


### categoricas

cb = list(combinations(categoricas,2))
p_value, modalidades, nombre_var = [], [], []

base2 = base.get(categoricas).copy()
base2["Fraude"] = base["Fraude"].copy()

for k in range(len(cb)):
     
    nombre_interaccion = cb[k][0]+"_"+cb[k][1]
    base2[nombre_interaccion] = base2[cb[k][0]].astype(str) + "_" + base2[cb[k][1]].astype(str)
    
    # Prueba chi cuadrado
    c1 = pd.DataFrame(pd.crosstab(base2["Fraude"],base2[nombre_interaccion]))
    pv = stats.chi2_contingency(c1)[1]
    
    # NÃºmero de modalidades por categorÃ­a
    mod_ = len(base2[nombre_interaccion].unique())
    
    # Guardar p value y modalidades
    if pv < 0.05 and mod_ < 40:
        nombre_var.append(nombre_interaccion)
        modalidades.append(mod_)
        p_value.append(pv)

pc = pd.DataFrame({'Variable':nombre_var,'Num Modalidades':modalidades,'p value':p_value})

seleccion1 = list(pc.loc[pc["p value"]<=0.05,"Variable"])
sel1 = base2.get(seleccion1)

lb1 = pd.get_dummies(base2.get(seleccion1), drop_first=True, prefix_sep="_", dtype=int)
lb1["Fraude"] = base["Fraude"].copy()


### cuanti-cualitativas
cat_cuanti = [(x,y) for x in cuantitativas for y in categoricas]

base_dum = pd.get_dummies(base[categoricas], drop_first=True, dtype=int)
prueba_dum = pd.get_dummies(prueba[categoricas], drop_first=True, dtype=int)

prueba_dum = prueba_dum.reindex(columns=base_dum.columns, fill_value=0)

# Listas para almacenar las variables de entrenamiento que pasen el filtro
var_mixtas_ganadoras = []

# --- 5.2 Evaluación en Entrenamiento ---
for num_col in cuantitativas:
    for dum_col in base_dum.columns:
        # Nombre único para la interacción mixta
        nombre_mixto = f"{num_col}_x_{dum_col}"
        
        # Multiplicación matemática directa: Numérica * Dummy (0 o 1)
        # Usamos los valores escalados si superaban los 1000 en el bloque de razones
        val_num_train = base[num_col] / 1000 if np.max(base[num_col]) > 1000 else base[num_col]
        vector_interaccion = val_num_train * base_dum[dum_col]
        
        # Prueba de Kruskal-Wallis sobre la interacción
        mue1 = vector_interaccion[base["Fraude"] == 1].to_numpy()
        mue2 = vector_interaccion[base["Fraude"] == 0].to_numpy()
        
        # Evitamos error si el vector se compone solo de ceros
        if len(np.unique(vector_interaccion)) > 1:
            p_val = stats.kruskal(mue1, mue2)[1]
            
            # FILTRO: Si aporta valor estadístico, guardamos el nombre de la columna
            if p_val < 0.05:
                var_mixtas_ganadoras.append((num_col, dum_col, nombre_mixto))

base_modelo5 = pd.DataFrame(index=base.index)
prueba_modelo5 = pd.DataFrame(index=prueba.index)

for num_col, dum_col, nombre_mixto in var_mixtas_ganadoras:
    # Construcción para Entrenamiento (Train)
    val_train = base[num_col] / 1000 if np.max(base[num_col]) > 1000 else base[num_col]
    base_modelo5[nombre_mixto] = val_train * base_dum[dum_col]
    
    # Construcción para Prueba (Test) - Sigue exactamente el molde de Train
    val_test = prueba[num_col] / 1000 if np.max(base[num_col]) > 1000 else prueba[num_col]
    prueba_modelo5[nombre_mixto] = val_test * prueba_dum[dum_col]

### variables cuantitativas al cuadrado
var_cuad = list(pcuadrado1["Variable2"])
X1 = base_cuadrado.get(var_cuad).copy()
y1 = base["Fraude"].astype(int)

modelo1 = XGBClassifier().fit(X1,y1)

importancias = modelo1.feature_importances_
imp1 = pd.DataFrame({'Variable':X1.columns,'Importancia':importancias})
imp1["Importancia"] = imp1["Importancia"] * 100 / np.sum(imp1["Importancia"])
imp1 = imp1.sort_values(["Importancia"],ascending=False).reset_index(drop=True)


#print(imp1)
### Interacciones
var_int = list(pxy["Variable"])
X2 = base_interacciones.get(var_int).copy()
y2 = base["Fraude"].astype(int)

modelo2 = XGBClassifier().fit(X2,y2)

importancias = modelo2.feature_importances_
imp2 = pd.DataFrame({'Variable':X2.columns,'Importancia':importancias})
imp2["Importancia"] = imp2["Importancia"] * 100 / np.sum(imp2["Importancia"])
imp2 = imp2.sort_values(["Importancia"],ascending=False).reset_index(drop=True)
#print(imp2)

### Razones
var_raz = list(prazones["Variable"])
X3 = base_razones1.get(var_raz).copy()
y3 = base["Fraude"].astype(int)

modelo3 = XGBClassifier().fit(X3,y3)

#print(len(list(X3.columns)))
#print(len(modelo3.feature_importances_))

importancias = modelo3.feature_importances_
imp3 = pd.DataFrame({'Variable':X3.columns,'Importancia':importancias})
imp3["Importancia"] = imp3["Importancia"] * 100 / np.sum(imp3["Importancia"])
imp3 = imp3.sort_values(["Importancia"],ascending=False).reset_index(drop=True)
#print(imp3.head())

### Categoricas

cov4 = [x for x in lb1.columns if x != "Fraude"]
lb1_train = lb1.get(cov4).copy()

X4 = lb1_train
y4 = base["Fraude"].astype(int)

modelo4 = XGBClassifier().fit(X4,y4)

importancias = modelo4.feature_importances_
imp4 = pd.DataFrame({'Variable':X4.columns,'Importancia':importancias})
imp4["Importancia"] = imp4["Importancia"] * 100 / np.sum(imp4["Importancia"])
imp4 = imp4.sort_values(["Importancia"],ascending=False).reset_index(drop=True)
#print(imp4.head(10))

## Cuanti-categóricas
X5 = base_modelo5.copy() 
y5 = base["Fraude"].astype(int)

modelo5 = XGBClassifier().fit(X5, y5)
imp5 = pd.DataFrame({'Variable': X5.columns, 'Importancia': modelo5.feature_importances_})
imp5["Importancia"] = (imp5["Importancia"] * 100) / np.sum(imp5["Importancia"])
imp5 = imp5.sort_values(["Importancia"], ascending=False).reset_index(drop=True)

def preparar_datos_evaluacion(df_nuevo):
    import pandas as pd
    import numpy as np
    
    D1_new = df_nuevo.get(cuantitativas).copy()
    
    D1_new["Balance_Tj_2"] = (D1_new["Balance_Tj"] / 1000) ** 2
    D1_new["Total_Transac_2"] = D1_new["Total_Transac"] ** 2
    
    D1_new["Balance_Tj_Total_Transac"] = (D1_new["Balance_Tj"] / 1000) * D1_new["Total_Transac"]
    D1_new["Num_Tarjetas_Balance_Tj"] = D1_new["Num_Tarjetas"] * (D1_new["Balance_Tj"] / 1000)
    D1_new["Num_Tarjetas_Total_Transac"] = D1_new["Num_Tarjetas"] * D1_new["Total_Transac"]
    
    D1_new["Num_Tarjetas_coc_Balance_Tj"] = D1_new["Num_Tarjetas"] / (D1_new["Balance_Tj"] + 0.01)
    D1_new["Total_Transac_coc_Num_Tarjetas"] = D1_new["Total_Transac"] / (D1_new["Num_Tarjetas"] + 0.01)
    D1_new["Num_Tarjetas_coc_Total_Transac"] = D1_new["Num_Tarjetas"] / (D1_new["Total_Transac"] + 0.01)
    
    vars_cat_sel = list(imp4.iloc[0:4, 0])
    vars_cat_cuant_sel = list(imp5.iloc[0:6, 0])

    seleccion1_limpia = [x for x in seleccion1 if x != "Fraude"]
    columnas_a_obtener = [x for x in seleccion1_limpia if x in df_nuevo.columns]
    
    if len(columnas_a_obtener) == 0:
        D4_new = pd.DataFrame(0, index=df_nuevo.index, columns=vars_cat_sel)
    else:
        
        dum_new = pd.get_dummies(df_nuevo.get(columnas_a_obtener), drop_first=True, prefix_sep="_", dtype=int)
        dum_new = dum_new.reindex(columns=lb1_train.columns, fill_value=0) 
        D4_new = dum_new.get(vars_cat_sel).copy()
        
        D4_new = D4_new.fillna(0).astype(int)
    
    
    base_dum_new = pd.get_dummies(df_nuevo[categoricas], drop_first=True, dtype=int)
    base_dum_new = base_dum_new.reindex(columns=base_dum.columns, fill_value=0) # <--- BLINDAJE AQUÍ
    
    df_mixto_new = pd.DataFrame(index=df_nuevo.index)
    for num_col, dum_col, nombre_mixto in var_mixtas_ganadoras:
        val_num = df_nuevo[num_col] / 1000 if np.max(df_nuevo[num_col]) > 1000 else df_nuevo[num_col]
        df_mixto_new[nombre_mixto] = val_num * base_dum_new[dum_col]
        
    D5_new = df_mixto_new.get(vars_cat_cuant_sel).copy()
    
    
    X_unificado = pd.concat([D1_new, D4_new, D5_new], axis=1)
    
    return X_unificado

def up_downsampling_replacement(X,y,set_seed):
    over = SMOTE(sampling_strategy='not majority',random_state=set_seed)
    under = RandomUnderSampler(sampling_strategy='not minority', replacement=True,random_state=set_seed)
    pipeline=Pipeline(steps=[('over',over),('under',under)])
    return pipeline.fit_resample(X,y)

def predicciones_train(modelo,X_train_Z,y_udr):
    yhat=modelo.predict(X_train_Z)
    y_prob=modelo.predict_proba(X_train_Z)[:, 1]
    
    # Matriz de confusión
    matriz_confusion=pd.DataFrame(confusion_matrix(y_udr, yhat))

    plt.figure(figsize=(7,5))
    graph_confusion=sns.heatmap(matriz_confusion,linewidth=0.01,annot=True,cmap='Blues',fmt='.0f')
    graph_confusion.set_title('Matriz de confusión para clasificación')
    plt.yticks(rotation=0)
    plt.xlabel('Valores predichos'); plt.ylabel('Valores observados')
    plt.show()
    
    auc_score = roc_auc_score(y_udr,y_prob)
    
    print('El área bajo la curva roc es: %0.5f' % auc_score)
    curva = BinaryClassification(y_udr, y_prob, labels=["class"])
    plt.figure(figsize=(8,6))
    curva.plot_roc_curve()
    plt.show()

def modelo_train_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state= 19052023
    )

    transformador1 = preprocessing.MinMaxScaler().fit(X_train)
    X_train_Z = transformador1.transform(X_train)
    X_test_Z = transformador1.transform(X_test)

    modelo1  = RandomForestClassifier(n_jobs=-1,max_features='sqrt', random_state=19052023)
    parametros1 = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 15, 20, None],     
        'min_samples_leaf': [2, 5, 10, 15],  
        'min_samples_split': [5, 10, 20]     
    }


    auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method='predict_proba')

    grilla1 = RandomizedSearchCV(
        estimator=modelo1,
        param_distributions=parametros1,
        n_iter=8,  
        scoring=auc_scorer,
        cv=5,
        n_jobs=-1,
        random_state=19052023
    )

    grilla1 = grilla1.fit(X_train_Z, y_train)

    print(grilla1.best_params_)

    predicciones_train(modelo = grilla1 ,X_train_Z = X_test_Z, y_udr = y_test)
    predicciones_train(modelo = grilla1 ,X_train_Z = X_train_Z, y_udr = y_train)

    prob_test = grilla1.best_estimator_.predict_proba(X_test_Z)[:, 1]
    auc_test = roc_auc_score(y_test, prob_test)
    print(f" AUC-ROC de Validación Interna: {auc_test:.5f}")




def modelo_test_train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state= 19052023
    )

    transformador2 = preprocessing.MinMaxScaler().fit(X_test)
    X_test_Z = transformador2.transform(X_test)
    X_train_Z = transformador2.transform(X_train)

    modelo2  = RandomForestClassifier(n_jobs=-1,max_features='sqrt', random_state=19052023)
    parametros1 = {
        'n_estimators': [200, 300, 400],
        'max_depth': [10, 15, 20, None],     
        'min_samples_leaf': [2, 5, 10, 15],  
        'min_samples_split': [5, 10, 20]     
    }

    auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method='predict_proba')
    
    grilla2 = RandomizedSearchCV(
    estimator=modelo2,
    param_distributions=parametros1,
        n_iter=8,  
        scoring=auc_scorer,
        cv=5,
        n_jobs=-1,
        random_state=19052023
    )

    grilla2 =grilla2.fit(X_test_Z, y_test)

    predicciones_train(modelo = grilla2 ,X_train_Z = X_test_Z, y_udr = y_test)
    predicciones_train(modelo = grilla2 ,X_train_Z = X_train_Z, y_udr = y_train)

    imp2 = pd.DataFrame({'Variable':X_test.columns,
              'Importancia':grilla2.best_estimator_.feature_importances_})
    imp2 = imp2.sort_values(["Importancia"],ascending=False)
    imp2.index = range(imp2.shape[0])
    print(imp2.head(5))

    

def modelo_sin_balanceo(X,y):
    
    T1 = preprocessing.MinMaxScaler().fit(X)
    X_Z = T1.transform(X)

    modelo_XG  = RandomForestClassifier(n_jobs=-1,max_features='sqrt', random_state=19052023, max_samples=0.5)
    parametros1 = {   
        'n_estimators': [100, 150, 200],      # 👈 Menos árboles en la grilla acelera la búsqueda drásticamente
        'max_depth': [10, 15, 20],            # Eliminamos 'None' que generaba árboles infinitos y pesados
        'min_samples_leaf': [5, 10, 20],  
        'min_samples_split': [10, 20]    
        }
    
    auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method='predict_proba')

    Random_BE = RandomizedSearchCV(
    estimator=modelo_XG,
    param_distributions=parametros1,
    n_iter=5,  
    scoring=auc_scorer,
    cv=3,
    n_jobs=-1,
    random_state=19052023
    )

    Random_BE = Random_BE.fit(X_Z, y)

    print(Random_BE.best_params_)

    predicciones_train(modelo = Random_BE ,X_train_Z = X_Z, y_udr = y)

    return Random_BE.best_estimator_, T1




def modelo_x_balanceado(X,y):
    X_udr, y_udr = up_downsampling_replacement(X,y,set_seed=20042021)

    T2 = preprocessing.MinMaxScaler().fit(X_udr)
    X_Z_udr = T2.transform(X_udr)

    modelo2_XG  = RandomForestClassifier(n_jobs=-1,max_features='sqrt', random_state=19052023, max_samples=0.5)
    parametros2 = {   
        'n_estimators': [100, 150, 200],      # 👈 Menos árboles en la grilla acelera la búsqueda drásticamente
        'max_depth': [10, 15, 20],            # Eliminamos 'None' que generaba árboles infinitos y pesados
        'min_samples_leaf': [5, 10, 20],  
        'min_samples_split': [10, 20]    
        }
    
    auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, response_method='predict_proba')
    
    Random_BE_Balanced = RandomizedSearchCV(
    estimator=modelo2_XG,
    param_distributions=parametros2,
    n_iter=5,  
    scoring=auc_scorer,
    cv=3,
    n_jobs=-1,
    random_state=19052023
    
    )

    Random_BE_Balanced = Random_BE_Balanced.fit(X_Z_udr, y_udr)

    print(Random_BE_Balanced.best_params_)

    predicciones_train(modelo = Random_BE_Balanced ,X_train_Z = X_Z_udr, y_udr = y_udr)

    return Random_BE_Balanced.best_estimator_, T2, X_udr.columns


base_train = base.copy()

base_modelo = preparar_datos_evaluacion(base_train.drop(["Fraude"], axis=1, errors='ignore'))
base_modelo["Y"] = base["Fraude"].astype(int)

X = base_modelo.drop(["Y"], axis=1)
y = base_modelo["Y"]

modelo_campeon, escalador_campeon, columnas_molde = modelo_x_balanceado(X,y)
modelo_sin_ba, escalador_sin_balanceo = modelo_sin_balanceo(X,y)

base_train = prueba.copy()

X_prueba_procesada = preparar_datos_evaluacion(base_train)
X_prueba_procesada = X_prueba_procesada.reindex(columns=columnas_molde, fill_value=0)

X_prueba_Z = escalador_campeon.transform(X_prueba_procesada)
X_prueba_Z_sin = escalador_sin_balanceo.transform(X_prueba_procesada)


evaluacion_prueba_balanceo = prueba[["ID"]].copy()
evaluacion_prueba_balanceo["Fraude"] = modelo_campeon.predict_proba(X_prueba_Z)[:, 1]

evaluacion_prueba_sin_balanceo = prueba[["ID"]].copy()
evaluacion_prueba_sin_balanceo["Fraude"] = modelo_sin_ba.predict_proba(X_prueba_Z_sin)[:, 1]


path2 = "output/"


evaluacion_prueba_sin_balanceo.to_csv(path2 + "submission_RForest_Sin_balanceo.csv", index=False)
evaluacion_prueba_balanceo.to_csv(path2 + "submission_RForest_Balanceo.csv", index=False)

