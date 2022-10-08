from io import BytesIO

import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, plot_roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from wandb.integration.xgboost import WandbCallback

import wandb
import model_card_toolkit as mctlib
import uuid
from datetime import date
import matplotlib.pyplot as plt
import base64
import seaborn as sns


def preprocess_data(data_folder):
    data = pd.read_csv(os.path.join(data_folder, "Telecom_customer churn.csv"))

    X = data.loc[:, data.columns != 'churn']
    y = data.loc[:, data.columns == 'churn']

    # Preprocessing
    X.drop('Customer_ID', axis=1, inplace=True)
    cat_columns = X.select_dtypes(exclude='number').columns
    num_columns = X.select_dtypes(include='number').columns

    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="mean")),
               ("scale", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    full_processor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, num_columns),
            ("categorical", categorical_pipeline, cat_columns),
        ]
    )

    X_processed = full_processor.fit_transform(X)
    y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
        y.values.reshape(-1, 1)
    )

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, stratify=y)
    return X_train, X_test, y_train, y_test


def wandb_model_train(X_train, X_test, y_train, y_test):
    default_params = {
        'objective': 'binary:logistic'
        , 'gamma': 1  ## def: 0
        , 'booster': 'gbtree'
        , 'learning_rate': 0.1  ## def: 0.1
        , 'max_depth': 3
        , 'min_child_weight': 100  ## def: 1
        , 'n_estimators': 100
        , 'nthread': 48
        , 'random_state': 42
        , 'reg_alpha': 0
        , 'reg_lambda': 0  ## def: 1
        , 'tree_method': 'hist'  # use `gpu_hist` to train on GPU
    }

    # Training
    model = xgb.XGBClassifier(**default_params, use_label_encoder=False, callbacks=[WandbCallback()])

    model.fit(X_train, y_train)
    f1 = f1_score(y_test, model.predict(X_test))
    logloss = log_loss(y_test, model.predict(X_test))
    wandb.log({'F1 score': f1})
    wandb.log({'LogLoss': logloss})
    print(f'F1 score: {f1}')
    print(f'LogLoss: {logloss}')
    wandb.finish()
    return model


def hyperparameter_tuning():
    default_params = {
        'objective': 'binary:logistic'
        , 'gamma': 1  ## def: 0
        , 'booster': 'gbtree'
        , 'learning_rate': 0.1  ## def: 0.1
        , 'max_depth': 3
        , 'min_child_weight': 100  ## def: 1
        , 'n_estimators': 100
        , 'nthread': 48
        , 'random_state': 42
        , 'reg_alpha': 0
        , 'reg_lambda': 0  ## def: 1
        , 'tree_method': 'hist'  # use `gpu_hist` to train on GPU
    }

    wandb.init(config=default_params)
    config = wandb.config
    # Initialize the XGBoostClassifier
    xgbmodel = xgb.XGBClassifier(**config)

    # Train the model, using the wandb_callback for logging
    xgbmodel.fit(X_train, y_train)

    preds = xgbmodel.predict(X_test)
    f1 = f1_score(y_test, preds)
    logloss = log_loss(y_test, preds)
    print(f"F1: {f1}")
    print(f"LogLoss: {logloss}")

    wandb.log({"F1 score": f1})
    wandb.log({"LogLoss": logloss})


def generate_model_card(path_to_results,
                        model,
                        x_train, x_test, y_train, y_test):
    def _plot_to_str():
        img = BytesIO()
        plt.savefig(img, format='png')
        return base64.encodebytes(img.getvalue()).decode('utf-8')

    mct = mctlib.ModelCardToolkit()

    model_card = mct.scaffold_assets()

    model_card.model_details.name = 'Telecom Churn Retention Prediction'
    model_card.model_details.overview = (
        'This model predicts whether customer will churn or not (i.e. supervised classification problem) based on '
        'a data, taken from Kaggle competition.')
    model_card.model_details.owners = [
        mctlib.Owner(name='Dmytro Pekach', contact='pekach.d@gmail.com')
    ]
    model_card.model_details.references = [
        mctlib.Reference(reference='https://www.kaggle.com/datasets/abhinav89/telecom-customer')
    ]
    model_card.model_details.version.name = str(uuid.uuid4())
    model_card.model_details.version.date = str(date.today())

    model_card.considerations.limitations = [mctlib.Limitation(description='Telecom churn retention prediction')]
    model_card.considerations.use_cases = [mctlib.UseCase(description='Telecom churn retention prediction')]
    model_card.considerations.users = [mctlib.User(description='Solution Architects'),
                                       mctlib.User(description='Data Scientists'),
                                       mctlib.User(description='ML Engineers')]

    model_card.quantitative_analysis.graphics.description = (
        'ROC curve and confusion matrix')

    plot_roc_curve(model, x_test, y_test)
    roc_curve = _plot_to_str()

    plot_confusion_matrix(model, x_test, y_test)
    confusion_matrix = _plot_to_str()

    model_card.quantitative_analysis.graphics.collection = [
        mctlib.Graphic(image=roc_curve),
        mctlib.Graphic(image=confusion_matrix)
    ]

    mct.update_model_card(model_card)
    mct.export_format(model_card, output_file=path_to_results)


if __name__ == '__main__':
    wandb.init(project="projector_mlops",
               entity="dimochak")

    ROOT_FOLDER = os.path.join(os.getcwd(), "..")
    DATA_FOLDER = os.path.join(ROOT_FOLDER, "data")
    MODEL_CARD_OUTPUT_PATH = os.path.join(ROOT_FOLDER, "results", "model_card.html")

    X_train, X_test, y_train, y_test = preprocess_data(DATA_FOLDER)
    model = wandb_model_train(X_train, X_test, y_train, y_test)

    # Hyperparameter search with W&B Sweep
    sweep_config = {
        "method": "random",
        "parameters": {
            "learning_rate": {
                "min": 0.001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.001,
                "max": 1.0
            },
            "max_depth": {
                "values": [3, 5, 7]
            },
            "min_child_weight": {
                "min": 1,
                "max": 150
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='projector_mlops')
    wandb.agent(sweep_id, hyperparameter_tuning, count=25)

    generate_model_card(MODEL_CARD_OUTPUT_PATH, model, X_train, X_test, y_train, y_test)
    quit(0)
