import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Conv1D, LSTM,  MaxPooling1D, BatchNormalization, Dense, Embedding, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import Recall, Precision

class NeuralNetworkModel:
    """

    example
    ------
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    from tensorflow.keras.metrics import Recall, Precision
    %load_ext tensorboard

    HYPERPARAMETERS = {
        "classification": True,
        "embedding": {
            "n_layers": 1,
            "n_hidden": [64],
            "n_dim": 4,
            "p_dropout": 0,
            "features": LABEL
        },
        "dense": {
            "n_layers": 1,
            "n_hidden": [32],
            "p_dropout": 0,
            "features": CONTINUOUS + ONE_HOT
        },
        "merged": {
            "n_layers": 2,
            "n_hidden": [512, 32],
            "p_dropout": 0,
        },
        "overall": {
            "activation": "relu",
            "l2_hidden": 10**-5,
            "output_activation": "sigmoid",
            "loss": "binary_crossentropy",
            "optimizer": tf.keras.optimizers.Adam(
                learning_rate=ExponentialDecay(
                    0.001,
                    decay_steps=10000,
                    decay_rate=0.9,
                    staircase=True
                ),
                epsilon=1e-07
            ),
            "metrics": ["accuracy"],
            "batch_size": 64,
            "epochs": 10
        },
        "callbacks": [
            tensorboard
        ]
    }
    """
    def __init__(self, params):
        """ """
        self.params=params
        self.model=None
        self.history=None

    def __log_model(self):
        mlflow.tensorflow.log_model(self.model, "nn model")

    def __log_inputs(self):
        pd.DataFrame(
            {"feature": self.params["embedding"]["features"]}
        ).to_csv('embedding_layer_inputs.csv')
        mlflow.log_artifact("embedding_layer_inputs.csv")
        pd.DataFrame(
            {"feature": self.params["dense"]["features"]}
        ).to_csv('dense_layer_inputs.csv')
        mlflow.log_artifact("dense_layer_inputs.csv")

    def __log_param(self):
        """

        """
        mlflow.log_params({
            # embedding
            "embedding_n_layers": self.params["embedding"]["n_layers"],
            "embedding_n_hidden": self.params["embedding"]["n_hidden"],
            "embedding_n_dim": self.params["embedding"]["n_dim"],
            "embedding_p_dropout": self.params["embedding"]["p_dropout"],
            # dense
            "dense_n_layers": self.params["dense"]["n_layers"],
            "dense_n_hidden": self.params["dense"]["n_hidden"],
            "embedding_p_dropout": self.params["dense"]["p_dropout"],
            # merged
            "merged_n_layers": self.params["merged"]["n_layers"],
            "merged_n_hidden": self.params["merged"]["n_hidden"],
            "merged_p_dropout": self.params["merged"]["p_dropout"],
            # overall
            "activation": self.params["overall"]["activation"],
            "l2_hidden": self.params["overall"]["l2_hidden"],
            "output_activation": self.params["overall"]["output_activation"],
            "loss": self.params["overall"]["loss"],
            "optimizer": self.params["overall"]["optimizer"],
            "metrics": self.params["overall"]["metrics"],
            "batch_size": self.params["overall"]["batch_size"],
            "epochs": self.params["overall"]["epochs"]
        })

    def __make_model(self, X_train):
        """ """
        # X_copy = pd.concat([X_train.copy(), X_train.copy()], ignore_index=True)
        X_copy = X_train.copy()

        # embedding layers
        if self.params["embedding"]["features"] is not None:
            emb_inp_layers = []
            emb_layers = []
            for feat in self.params["embedding"]["features"]:
                emb_input_dim = X_copy[feat].nunique() + 1
                inp_layer = layers.Input(shape=(1,), name=f"input_{feat}")
                emb_inp_layers.append(inp_layer)
                emb_output_dim=int(min(
                    self.params["embedding"]["n_dim"], X_copy[feat].nunique() / 2
                ))
                emb_layer = layers.Embedding(
                    emb_input_dim,
                    emb_output_dim,
                    input_length=1,
                    name=f"embed_{feat}"
                )(inp_layer)
                emb_layer = layers.Dropout(self.params["embedding"]["p_dropout"])(emb_layer)
                emb_layer = layers.Flatten()(emb_layer)
                emb_layers.append(emb_layer)
            if len(self.params["embedding"]["features"]) > 1:
                emb_merged = layers.Concatenate()(emb_layers)
            else:
                emb_inp_layers = emb_inp_layers[0]
                emb_merged = emb_layers[0]
            for i in range(self.params["embedding"]["n_layers"]):
                emb_merged = layers.Dense(
                    self.params["embedding"]["n_hidden"][i],
                    activation=self.params["overall"]["activation"],
                    kernel_regularizer=l2(self.params["overall"]["l2_hidden"]),
                    name=f"embed_dense_{i+1}"
                )(emb_merged)
                emb_merged = layers.Dropout(
                    self.params["embedding"]["p_dropout"]
                )(emb_merged)

        # dense layers
        if self.params["dense"]["features"] is not None:
            dense_input_dim = len(self.params["dense"]["features"])
            dense_input_layers = layers.Input(shape=(dense_input_dim,), name="dense_input")
            for i in range(self.params["dense"]["n_layers"]):
                if i == 0:
                    dense = layers.Dense(
                        self.params["dense"]["n_hidden"][i],
                        activation=self.params["overall"]["activation"],
                        kernel_regularizer=l2(self.params["overall"]["l2_hidden"]),
                        name="dense_1"
                    )(dense_input_layers)
                    dense = layers.Dropout(
                        self.params["dense"]["p_dropout"],
                        name="dense_dropout_1"
                    )(dense)
                else:
                    dense = layers.Dense(
                        self.params["dense"]["n_hidden"][i],
                        activation=self.params["overall"]["activation"],
                        kernel_regularizer=l2(self.params["overall"]["l2_hidden"]),
                        name=f"dense_{1+1}"
                    )(dense)
                    dense = layers.Dropout(
                        self.params["dense"]["p_dropout"],
                        name=f"dense_dropout_{1+1}"
                    )(dense)

        # merged layer
        if self.params["embedding"]["features"] is None:
            if self.params["dense"]["features"] is None:
                print("No input features are selected.")
                return None
            else:
                merged = dense
        else:
            if self.params["dense"]["features"] is None:
                merged = emb_merged
            else:
                merged = layers.Concatenate()([emb_merged, dense])

        for i in range(self.params["merged"]["n_layers"]):
            merged = layers.Dense(
                self.params["merged"]["n_hidden"][i],
                activation=self.params["overall"]["activation"],
                kernel_regularizer=l2(self.params["overall"]["l2_hidden"]),
                name=f"merged_dense_{i+1}"
            )(merged)
            merged = layers.Dropout(
                self.params["merged"]["p_dropout"],
                name=f"merged_dropout_{i+1}"
            )(merged)

        output = layers.Dense(
            1,
            activation=self.params["overall"]["output_activation"],
            name="output"
        )(merged)
        self.model = Model(inputs=[emb_inp_layers, dense_input_layers], outputs=output)
        self.model.compile(
            loss=self.params["overall"]["loss"],
            optimizer=self.params["overall"]["optimizer"],
            metrics=self.params["overall"]["metrics"]
        )
        # print(self.model.summary())

    def __log_metrics(self, prefix, y_true, y_proba, w=None):
        """

        """
        y_pred = y_proba.round(decimals=0)
        if self.params["classification"] is True:
            # accuracy
            acc = round(accuracy_score(y_true, y_pred, sample_weight=w), 2)
            # precision
            pre = round(precision_score(y_true, y_pred, sample_weight=w), 2)
            # recall
            rec = round(recall_score(y_true, y_pred, sample_weight=w), 2)
            # auc score
            auc = round(roc_auc_score(y_true, y_proba, sample_weight=w), 2)
            # f1_score
            f1s = round(f1_score(y_true, y_pred, sample_weight=w), 2)

            mlflow.log_metrics({
                f"{prefix}_accuracy": acc,
                f"{prefix}_precision": pre,
                f"{prefix}_recall": rec,
                f"{prefix}_auc": auc,
                f"{prefix}_f1_score": f1s
            })

            print(prefix)
            print(f"\t Accuracy: {acc}")
            print(f"\t Precision: {pre}")
            print(f"\t Recall: {rec}")
            print(f"\t AUC: {auc}")
            print(f"\t F1 Score: {f1s}")


    def fit(self, X_train, X_test, y_train, y_test, w_train=None, w_test=None):
        """

        """
        self.__make_model(X_train)
        self.__log_inputs()
        with mlflow.start_run(nested=True):
            self.__log_param()
            self.history = self.model.fit(
                [
                    *[X_train[col].to_numpy() for col in self.params["embedding"]["features"]],
                    X_train[[col for col in X_train.columns if col in self.params["dense"]["features"]]].to_numpy()
                ],
                np.reshape(y_train.to_numpy(), (-1, 1)),
                sample_weight=w_train,
                epochs=HYPERPARAMETERS["overall"]["epochs"],
                batch_size=HYPERPARAMETERS["overall"]["batch_size"],
                validation_data = (
                    [
                        *[X_test[col].to_numpy() for col in self.params["embedding"]["features"]],
                        X_test[[col for col in X_test.columns if col in self.params["dense"]["features"]]].to_numpy()
                    ],
                    np.reshape(y_test.to_numpy(), (-1, 1)),
                    w_test
                ),
                verbose=0,
                callbacks=self.params["callbacks"]
            )
            train_proba = self.model.predict([
                *[X_train[col].to_numpy() for col in self.params["embedding"]["features"]],
                X_train[[col for col in X_train.columns if col in self.params["dense"]["features"]]].to_numpy()
            ])
            self.__log_metrics("train", y_train, train_proba)
            test_proba = self.model.predict([
                *[X_test[col].to_numpy() for col in self.params["embedding"]["features"]],
                X_test[[col for col in X_test.columns if col in self.params["dense"]["features"]]].to_numpy()
            ])
            self.__log_metrics("test", y_test, test_proba)
            # self.__log_model()

    def feature_importance(self, X_train, y_train, w_train=None):
        """ """
        eval = self.model.evaluate([
                *[X_train[col].to_numpy() for col in self.params["embedding"]["features"]],
                X_train[[col for col in X_train.columns if col in self.params["dense"]["features"]]].to_numpy()
            ],
            np.reshape(y_train.to_numpy(), (-1, 1)),
            batch_size=X_train.shape[0],
            verbose=0
        )
        baseline_val_metrics = eval[1]

        features = self.params["embedding"]["features"] + self.params["dense"]["features"]
        importance = {feat: None for feat in features}
        for feat in features:
            X_copy = X_train.copy()
            X_copy[feat] = X_train[feat].sample(frac=1).values

            eval = self.model.evaluate([
                    *[X_copy[col].to_numpy() for col in self.params["embedding"]["features"]],
                    X_copy[[col for col in X_copy.columns if col in self.params["dense"]["features"]]].to_numpy()
                ],
                np.reshape(y_train.to_numpy(), (-1, 1)),
                batch_size=X_train.shape[0],
                verbose=0
            )
            val_metrics = eval[1]

            importance[feat] = abs(baseline_val_metrics - val_metrics)

        # norm_const = 1.0 / sum(importance.itervalues())
        # importance = {k: v*norm_const for k, v in importance.iteritems()}
        # X_importance = pd.DataFrame.from_dict(importance,  orient=columns, columns=["feature", "importance"])
        # return X_importance
        return importance

    # def predict_proba(self, X):
    #     """ """
    #     prob = self.model.predict([
    #         *[X[col].to_numpy() for col in self.params["embedding"]["features"]],
    #         X[[col for col in X.columns if col in self.params["dense"]["features"]]].to_numpy()
    #     ])
    #     return prob
