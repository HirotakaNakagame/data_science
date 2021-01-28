import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class Featurizer:
    """
    this featurizer object fit/apply one-hot-encoder, label-encoder, standard-scaler,
    and nn-embedding-encoder
    Args:
        one_hot_columns (list):
        label_encoder_columns (list):
        scaler_columns (list):
        nn_columns (list):
        nn_encoder_hyperparameters (list):

    Attributes:
        one_hot_encoders ():
        label_encoding_dict ():
        scalers ():
        nn_encoders ():

    """
    def __init__(
            self, one_hot_columns=None, label_encoder_columns=None, scaler_columns=None,
            nn_columns=None, nn_encoder_hyperparameters=None, drop_original=False
    ):
        self.one_hot_columns = one_hot_columns
        self.label_encoder_columns = label_encoder_columns
        self.scaler_columns = scaler_columns
        self.nn_columns = nn_columns

        self.nn_encoder_hyperparameters = nn_encoder_hyperparameters
        self.drop_original = drop_original

        self.one_hot_encoders = None
        self.label_encoders = None
        self.label_encoding_dict = {}
        self.scalers = None
        self.nn_encoders = None

    def __fit_one_hot_encoders(self, X):
        """
        fit one hot encoder on `one_hot_columns`
        Args:
            X (pd.DataFrame): dataset to fit one-hot encoder on
        """
        self.one_hot_encoders = OneHotEncoder(handle_unknown='ignore')
        self.one_hot_encoders.fit(X[self.one_hot_columns])

    def __transform_one_hot_encoders(self, X):
        """
        apply one hot encoders on every `one_hot_columns`
        Args:
            X (pd.DataFrame): dataset to apply one hot encoders on
        Returns:
            pd.DataFrame
        """
        X_one_hot_encoded = pd.DataFrame(
            self.one_hot_encoders.transform(X[self.one_hot_columns]).toarray()
            )
        X_one_hot_encoded.columns = self.one_hot_encoders.get_feature_names(
            self.one_hot_columns
            )
        X = X.merge(X_one_hot_encoded, left_index=True, right_index=True, how="inner")
        if self.drop_original is True:
            X.drop(self.one_hot_columns, axis=1, inplace=True)
        return X

    def __fit_label_encoder(self, series):
        """
        fit a label encoder for a column
        Args:
            series (pd.Series): column to fit label encoder on
        """
        col_name = series.name
        self.label_encoding_dict[col_name] = {}
        for i, value in enumerate(series.unique()):
            self.label_encoding_dict[col_name][value] = i
        def label_encoder(series):
            col_name = series.name
            series = series.map(self.label_encoding_dict[col_name])
            series.fillna(len(self.label_encoding_dict[col_name])+1, inplace=True)
            return series
        return label_encoder

    def __fit_label_encoders(self, X):
        """
        fit a label encoder for `label_encoder_columns`
        Args:
            X (pd.DataFrame): dataset to fit label encoder on
        """
        self.label_encoders = {}
        for label_encoder_column in self.label_encoder_columns:
            self.label_encoders[label_encoder_column] = self.__fit_label_encoder(
                X[label_encoder_column]
            )

    def __transform_label_encoder(self, series):
        """
        apply a label encoder for a column
        Args:
            series (pd.Series): column to apply label encoder on
        Returns:
            pd.Series
        """
        return self.label_encoders[series.name](series)


    def __transform_label_encoders(self, X):
        """
        apply label encoders on every `label_encoder_columns`
        Args:
            X (pd.DataFrame): dataset to apply label encoders on
        Returns:
            pd.DataFrame
        """
        for label_encoder_column in self.label_encoder_columns:
            X[label_encoder_column] = self.__transform_label_encoder(X[label_encoder_column])
        return X

    def __fit_scalers(self, X):
        """
        fit standard scaler on `scaler_columns`
        Args:
            X (pd.DataFrame): dataset to fit scaler on
        """
        self.scalers = StandardScaler()
        self.scalers.fit(X[self.scaler_columns])

    def __transform_scalers(self, X):
        """
        apply standard scaler on `scaler_columns`
        Args:
            X (pd.DataFrame): dataset to apply scaler on
        """
        X[self.scaler_columns] = self.scalers.transform(X[self.scaler_columns])
        return X


    def fit(self, X):
        """
        fit one-hot-encoder, label-encoder, scaler, and nn-encoder
        Args:
            X (pd.DataFrame): dataset to fit all the featurizer on
        """
        X_copy = X.copy()
        if self.one_hot_columns is not None:
            print("Fitting One Hot Encoder")
            self.__fit_one_hot_encoders(X_copy)

        if self.label_encoder_columns is not None:
            print("Fitting Label Encoder")
            self.__fit_label_encoders(X_copy)

        if self.scaler_columns is not None:
            print("Fitting Scaler")
            self.__fit_scalers(X_copy)


    def transform(self, X):
        """
        apply every featurized that was fit in `fit`
        Args:
            X (pd.DataFrame): dataset to apply featurizer on
        Return:
            pd.DataFrame
        """
        if self.one_hot_encoders is not None:
            print("One Hot Encoding")
            X = self.__transform_one_hot_encoders(X)

        if self.label_encoders is not None:
            print("Label Encoding")
            X = self.__transform_label_encoders(X)

        if self.scalers is not None:
            print("Scaling")
            X = self.__transform_scalers(X)

        return X
