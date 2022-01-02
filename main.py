from clustering_selection import ClusteringSelection
from decision_dependent_direct_knn import DecisionDependentDirectKNN

from decision_dependent_distance_based_knn import DecisionDependentDistanceBasedKNN
from decision_independent_direct_knn import DecisionIndependentDirectKNN
from decision_independent_distance_based_knn import DecisionIndependentDistanceBasedKNN

from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder


def iris_preprocessing(dataframe: pd.DataFrame) -> tuple:
    data = dataframe.data
    target = dataframe.target
    return data, target


def preprocessing_for_bodyPreformance(dataframe: pd.DataFrame) -> tuple:
    """
    :param dataframe:  bodyPreformance dataframe
    :return: touple of X and Y after preprocessing
    """
    data = dataframe.copy()
    # drop none values from dataset
    data = data.dropna(axis=0)
    df_target_column_name = 'class'
    data_traget = data[df_target_column_name]

    # define y to categories instead of string (A-D)
    data_traget = data_traget.astype("category").cat.codes
    data.drop(columns=df_target_column_name, inplace=True)

    # hotspot for gender column
    gender = pd.get_dummies(data['gender'])
    data.drop(columns='gender', inplace=True)
    data = data.join(gender)

    return data, data_traget


def music_genre_preprocessing(dataframe: pd.DataFrame) -> tuple:
    # todo
    return ()


def marketing_preprocess_function(dataframe: pd.DataFrame) -> tuple:
    """
    :param dataframe:  marketing_campaign dataframe
    :return: touple of X and Y after preprocessing
    """
    dataframe = dataframe.copy()
    # split data to X and y
    target_column_name = 'Marital_Status'

    # Change all "Alone" Status to Single
    dataframe[target_column_name] = dataframe[target_column_name].apply(lambda x: "Single" if x == "Alone" else x)
    df_target = dataframe[target_column_name]
    dataframe.drop(columns=target_column_name, inplace=True)

    # change y data to categories instead of strings
    df_target = df_target.astype("category")
    df_target = df_target.cat.codes

    # split data to bins
    dataframe['Income'].fillna(dataframe['Income'].mean(), inplace=True)
    dataframe['Income'] = pd.to_numeric(dataframe['Income'], errors='coerce')
    dataframe['Income'] = pd.qcut(dataframe['Income'], q=4, labels=[0, 1, 2, 3])
    dataframe['Year_Birth'] = pd.qcut(dataframe['Year_Birth'], q=4, labels=[0, 1, 2, 3])

    # hotspot for education column
    education = pd.get_dummies(dataframe['Education'])
    dataframe = dataframe.join(education)
    dataframe.drop(columns='Education', inplace=True)

    # Drop Id coloumn
    dataframe.drop(columns='ID', inplace=True)

    # Drop Dt_Customer, data format that didnt help for classification
    dataframe.drop(columns='Dt_Customer', inplace=True)

    return dataframe, df_target


def print_metrics(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    Using for print metrics after using model.fit
    we use accuracy, f1 and precision metrics for evaluation, more details found in attached DOC
    :param model: classification model that we want to evaluate
    :param X_test: X data for test as dataframe
    :param y_test: label for each row of X_test, as dataframe
    :return:
    """
    print(f'accuracy_score is : {accuracy_score(y_test, model.predict(X_test))}')


def decision_dependent_direct_knn_test(dataframe: pd.DataFrame, preprocess_function) -> None:
    """
    This function use `decision_dependent_direct_knn_test` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dddk = DecisionDependentDirectKNN()
    dddk.fit(X_train, y_train)
    print_metrics(model=dddk, X_test=X_test, y_test=y_test)


def decision_dependent_distance_based_knn_test(dataframe: pd.DataFrame, preprocess_function) -> None:
    """
    This function use `decision_dependent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    dddk = DecisionDependentDistanceBasedKNN()
    dddk.fit(X_train, y_train)
    print_metrics(model=dddk, X_test=X_test, y_test=y_test)


def decision_independent_direct_knn_test(dataframe: pd.DataFrame, preprocess_function) -> None:
    """
    This function use `decision_independent_direct_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    didk = DecisionIndependentDirectKNN()
    didk.fit(X_train, y_train)
    print_metrics(model=didk, X_test=X_test, y_test=y_test)


def decision_independent_distance_based_knn_test(dataframe: pd.DataFrame, preprocess_function) -> None:
    """
    This function use `decision_independent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    didbk = DecisionIndependentDistanceBasedKNN()
    didbk.fit(X_train, y_train)
    print_metrics(model=didbk, X_test=X_test, y_test=y_test)


def clustering_selection_test(dataframe: pd.DataFrame, preprocess_function) -> None:
    """
    This function use `decision_independent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    didbk = ClusteringSelection()
    didbk.fit(X_train, y_train)
    print_metrics(model=didbk, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    iris = load_iris()
    decision_dependent_direct_knn_test(iris, iris_preprocessing)
    # print("music genre")
    # decision_dependent_direct_knn_test(pd.read_csv('data/tiktok.csv'), music_genre_preprocessing)
    # print("-" * 100)
    # decision_dependent_distance_based_knn_test(pd.read_csv('data/tiktok.csv'), music_genre_preprocessing)
    # print("-" * 100)
    # decision_independent_direct_knn_test(pd.read_csv('data/tiktok.csv'), music_genre_preprocessing)
    # print("-" * 100)
    # decision_independent_distance_based_knn_test(pd.read_csv('data/tiktok.csv'), music_genre_preprocessing)
    # print("-" * 100)
    # clustering_selection_test(pd.read_csv('data/tiktok.csv'), music_genre_preprocessing)
    # print("\n\n")
    # print("-" * 100)
    # print("marketing_campaign")
    # decision_dependent_direct_knn_test(pd.read_csv(), marketing_preprocess_function)
    # print("-" * 100)
    # decision_dependent_distance_based_knn_test(pd.read_csv(), marketing_preprocess_function)
    # print("-" * 100)
    # decision_independent_direct_knn_test(pd.read_csv(), marketing_preprocess_function)
    # print("-" * 100)
    # decision_independent_distance_based_knn_test(pd.read_csv(), marketing_preprocess_function)
    # print("-" * 100)
    # clustering_selection_test(pd.read_csv(), marketing_preprocess_function)
    # print("\n\n")
    # print("bodyPerformance")
    # decision_dependent_direct_knn_test(pd.read_csv(), preprocessing_for_bodyPreformance)
    # print("-" * 100)
    # decision_dependent_distance_based_knn_test(pd.read_csv(), preprocessing_for_bodyPreformance)
    # print("-" * 100)
    # decision_independent_direct_knn_test(pd.read_csv(), preprocessing_for_bodyPreformance)
    # print("-" * 100)
    # decision_independent_distance_based_knn_test(pd.read_csv(), preprocessing_for_bodyPreformance)
    # print("-" * 100)
    # clustering_selection_test(pd.read_csv(), preprocessing_for_bodyPreformance)
    # print("\n\n")
