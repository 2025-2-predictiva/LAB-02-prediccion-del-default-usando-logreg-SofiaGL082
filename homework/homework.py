# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline as sk_make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

def read_data(file):
    dir = Path("files/input")
    return pd.read_csv(dir / file, compression="zip")

def clean_data(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.drop(columns="ID")
    df = df.dropna()
    df = df[df["MARRIAGE"] != 0]
    df = df[df["EDUCATION"] != 0]

    def education(e):
        if e > 4:
            return 4
        return e
    df["EDUCATION"] = df["EDUCATION"].map(education)
    df = df.dropna()
    return df


def make_xy(df: pd.DataFrame):
    x = df[df.columns].copy()
    x.drop(columns="default", inplace=True)
    y = df["default"].copy()
    return x, y


def make_train_test_split(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split

    x, y = make_xy(df)

    (x_train, x_test, y_train, y_test) = train_test_split(
        x,
        y,
        random_state=0,
    )
    return x_train, x_test, y_train, y_test


def make_pipeline():
    return sk_make_pipeline(
        ColumnTransformer(
            transformers=[("encoder", OneHotEncoder(handle_unknown="ignore"), ['SEX', 'EDUCATION', 'MARRIAGE'])],
            remainder=MinMaxScaler()
        ),
        SelectKBest(),
        LogisticRegression(penalty="l2")
    )

def make_grid_search(estimator, param_grid, cv=10):
    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

def save_estimator(estimator):
    import pickle
    import gzip
    from pathlib import Path

    outdir = Path("files/models")
    outdir.mkdir(exist_ok=True)

    with gzip.open(outdir / "model.pkl.gz", "wb") as file:
        pickle.dump(estimator, file)

def save_metrics(metrics):
    import json
    from pathlib import Path

    outdir = Path("files/output")
    outdir.mkdir(exist_ok=True)

    strmetrics = []
    for metric in metrics:
        strmetrics.append(json.dumps(metric)+"\n")

    with open(outdir / "metrics.json", "w") as file:
        file.writelines(strmetrics)

def train_estimator(x_train, y_train):
    pipeline = make_pipeline()
    estimator = make_grid_search(pipeline, param_grid={
        "selectkbest__k": range(1, 11),
        "logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
    })

    estimator.fit(x_train, y_train)
    save_estimator(estimator)
    return estimator

def eval_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score

    acc = precision_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, bacc, recall, f1

def eval_confusion(dataset, y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    m = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {
            "predicted_0": int(m[0][0]),
            "predicted_1": int(m[0][1])
        },
        "true_1": {
            "predicted_0": int(m[1][0]),
            "predicted_1": int(m[1][1])
        }
    }

def main():
    x_train, y_train = make_xy(clean_data(read_data("train_data.csv.zip")))
    x_test, y_test = make_xy(clean_data(read_data("test_data.csv.zip")))

    estimator = train_estimator(x_train, y_train)

    metrics = []
    for dataset, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = estimator.predict(x)
        acc, bacc, recall, f1 = eval_metrics(y, y_pred)
        metrics.append({
            "type": "metrics",
            "dataset": dataset,
            "precision": acc,
            "balanced_accuracy": bacc,
            "recall": recall,
            "f1_score": f1
        })

    cm_matrix = []
    for dataset, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
        y_pred = estimator.predict(x)
        cm_matrix.append(eval_confusion(dataset, y, y_pred))

    save_metrics([*metrics, *cm_matrix])

main()