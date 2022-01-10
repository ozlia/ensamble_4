from clustering_selection import ClusteringSelection
from decision_dependent_direct_knn import DecisionDependentDirectKNN
import time
from decision_dependent_distance_based_knn import DecisionDependentDistanceBasedKNN
from decision_independent_direct_knn import DecisionIndependentDirectKNN
from decision_independent_distance_based_knn import DecisionIndependentDistanceBasedKNN
from sklearn.metrics import accuracy_score, precision_score, f1_score

import pandas as pd
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')


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
    """

    :param dataframe: music_genre dataframe
    :return: tuple of X and Y after preprocessing
    """
    dataframe = dataframe.copy()

    # Drop useless columns
    dataframe.drop(columns=['artist_name', 'track_name', 'obtained_date', 'tempo', 'instance_id'], inplace=True)
    dataframe.dropna(inplace=True)
    print (len(dataframe))

    # split data to X and y
    target_column_name = 'music_genre'

    df_target = dataframe[target_column_name]
    dataframe.drop(columns=target_column_name, inplace=True)

    # change y data to categories instead of strings
    df_target = df_target.astype("category")
    df_target = df_target.cat.codes



    # split columns to bins:
    dataframe[dataframe['duration_ms'] == -1] = dataframe[dataframe['duration_ms'] != -1]['duration_ms'].mean()
    dataframe['duration_ms'] = pd.qcut(dataframe['duration_ms'], q=4, labels=[0, 1, 2, 3])

    # hotspot for mode
    mode = pd.get_dummies(dataframe['mode'])
    dataframe = dataframe.join(mode, lsuffix="mode_")
    dataframe.drop(columns='mode', inplace=True)

    # hotspot for key
    key = pd.get_dummies(dataframe['key'])
    dataframe = dataframe.join(key, lsuffix="key_")
    dataframe.drop(columns='key', inplace=True)

    return dataframe, df_target


def bank_preprocess_function(dataframe: pd.DataFrame) -> tuple:
    """
    :param dataframe:  bank dataframe
    :return: tuple of X and Y after preprocessing
    """
    dataframe = dataframe.copy()
    # split data to X and y
    target_column_name = 'marital'

    df_target = dataframe[target_column_name]
    dataframe.drop(columns=target_column_name, inplace=True)

    # change y data to categories instead of strings
    df_target = df_target.astype("category")
    df_target = df_target.cat.codes

    # split data to bins
    dataframe['balance'].fillna(dataframe['balance'].mean(), inplace=True)
    dataframe['balance'] = pd.to_numeric(dataframe['balance'], errors='coerce')
    dataframe['balance'] = pd.qcut(dataframe['balance'], q=4, labels=[0, 1, 2, 3])

    dataframe['age'].fillna(dataframe['age'].mean(), inplace=True)
    dataframe['age'] = pd.to_numeric(dataframe['age'], errors='coerce')
    dataframe['age'] = pd.qcut(dataframe['age'], q=4, labels=[0, 1, 2, 3])

    # hotspot for education column
    education = pd.get_dummies(dataframe['education'])
    dataframe = dataframe.join(education, lsuffix="edu_")
    dataframe.drop(columns='education', inplace=True)

    # hotspot for jobs
    jobs = pd.get_dummies(dataframe['job'])
    dataframe = dataframe.join(jobs, lsuffix="job_")
    dataframe.drop(columns='job', inplace=True)

    # hotspot for default
    default = pd.get_dummies(dataframe['default'])
    dataframe = dataframe.join(default, lsuffix="default_")
    dataframe.drop(columns='default', inplace=True)

    # hotspot for loan
    loan = pd.get_dummies(dataframe['loan'])
    dataframe = dataframe.join(loan, lsuffix="loan_")
    dataframe.drop(columns='loan', inplace=True)

    # hotspot for housing
    housing = pd.get_dummies(dataframe['housing'])
    dataframe = dataframe.join(housing, lsuffix="house_")
    dataframe.drop(columns='housing', inplace=True)

    # hotspot for deposit
    deposit = pd.get_dummies(dataframe['deposit'])
    dataframe = dataframe.join(deposit, lsuffix="deposit_")
    dataframe.drop(columns='deposit', inplace=True)

    # Drop contact coloumn - nonsanse values
    dataframe.drop(columns='contact', inplace=True)

    # Drop day and month, data and poutcome that didnt help for classification
    dataframe.drop(columns=['day', 'month', 'poutcome'], inplace=True)
    print (len(dataframe))
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
    model_prediction = model.predict(X_test)
    print(f'accuracy_score is : {accuracy_score(y_test, model_prediction)}')
    print(f'f1_score score is : {f1_score(y_test, model_prediction, average="weighted")}')
    print(f'precision score is : {precision_score(y_test, model_prediction, average="weighted")}')


def decision_dependent_direct_knn_test(dataframe: pd.DataFrame, preprocess_function) -> list:
    """
    This function use `decision_dependent_direct_knn_test` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return: trainded models list
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    dddk = DecisionDependentDirectKNN()
    dddk.fit(X_train, y_train)
    print_metrics(model=dddk, X_test=X_test, y_test=y_test)
    return dddk.export_trained_models_as_list()


def decision_dependent_distance_based_knn_test(dataframe: pd.DataFrame, preprocess_function, pre_traind_models: list = None) -> None:
    """
    This function use `decision_dependent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    dddk = DecisionDependentDistanceBasedKNN()
    need_models_fiting = pre_traind_models is not None
    if pre_traind_models is not None:
        dddk.import_models_from_list(pre_traind_models)
    dddk.fit(X_train, y_train, need_models_fiting)
    print_metrics(model=dddk, X_test=X_test, y_test=y_test)


def decision_independent_direct_knn_test(dataframe: pd.DataFrame, preprocess_function, pre_traind_models: list = None) -> None:
    """
    This function use `decision_independent_direct_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    didk = DecisionIndependentDirectKNN()
    need_models_fiting = pre_traind_models is not None
    if pre_traind_models is not None:
        didk.import_models_from_list(pre_traind_models)
    didk.fit(X_train, y_train, need_models_fiting)
    print_metrics(model=didk, X_test=X_test, y_test=y_test)


def decision_independent_distance_based_knn_test(dataframe: pd.DataFrame, preprocess_function, pre_traind_models: list = None) -> None:
    """
    This function use `decision_independent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    didbk = DecisionIndependentDistanceBasedKNN()
    need_models_fiting = pre_traind_models is not None
    if pre_traind_models is not None:
        didbk.import_models_from_list(pre_traind_models)
    didbk.fit(X_train, y_train, need_models_fiting)
    print_metrics(model=didbk, X_test=X_test, y_test=y_test)


def clustering_selection_test(dataframe: pd.DataFrame, preprocess_function, pre_traind_models: list = None) -> None:
    """
    This function use `decision_independent_distance_based_knn` model for evaluate performance of multi classification task
    :param dataframe: dataframe
    :param preprocess_function: preprocess function for given dataframe
    :return:
    """
    X, y = preprocess_function(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    didbk = ClusteringSelection()
    need_models_fiting = pre_traind_models is not None
    if pre_traind_models is not None:
        didbk.import_models_from_list(pre_traind_models)
    didbk.fit(X_train, y_train, need_models_fiting)
    print_metrics(model=didbk, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    # bank_df = pd.read_csv(r'datasets/music_genre.csv')
    # print (bank_df['job'].unique())
    # music_genre_preprocessing(bank_df)

    # iris = load_iris()
    # decision_dependent_direct_knn_test(iris, iris_preprocessing)
    # decision_dependent_distance_based_knn_test(iris, iris_preprocessing)
    # decision_independent_direct_knn_test(iris, iris_preprocessing)
    # decision_independent_distance_based_knn_test(iris, iris_preprocessing)

    # print("music genre")
    # t = time.time()
    # pre_trained_list = decision_dependent_direct_knn_test(pd.read_csv('datasets/music_genre.csv').sample(n= 50), music_genre_preprocessing)
    # print("-" * 100)
    # decision_dependent_distance_based_knn_test(pd.read_csv('datasets/music_genre.csv'), music_genre_preprocessing, pre_trained_list)
    # print("-" * 100)
    # decision_independent_direct_knn_test(pd.read_csv('datasets/music_genre.csv'), music_genre_preprocessing, pre_trained_list)
    # print("-" * 100)
    # decision_independent_distance_based_knn_test(pd.read_csv('datasets/music_genre.csv'), music_genre_preprocessing, pre_trained_list)
    # print("-" * 100)
    # clustering_selection_test(pd.read_csv('datasets/music_genre.csv'), music_genre_preprocessing, pre_trained_list)
    # print(time.time() - t)
    # print("\n\n")
    # print("-" * 100)
    print("bank")
    pre_trained_list = decision_dependent_direct_knn_test(pd.read_csv(r'datasets/bank.csv'), bank_preprocess_function)
    print("-" * 100)
    decision_dependent_distance_based_knn_test(pd.read_csv(r'datasets/bank.csv'), bank_preprocess_function, pre_trained_list)
    print("-" * 100)
    decision_independent_direct_knn_test(pd.read_csv(r'datasets/bank.csv'), bank_preprocess_function, pre_trained_list)
    print("-" * 100)
    decision_independent_distance_based_knn_test(pd.read_csv(r'datasets/bank.csv'), bank_preprocess_function, pre_trained_list)
    print("-" * 100)
    clustering_selection_test(pd.read_csv(r'datasets/bank.csv'), bank_preprocess_function, pre_trained_list)
    print("\n\n")
    # print("bodyPerformance")
    # pre_trained_list = decision_dependent_direct_knn_test(pd.read_csv(r'datasets/bodyPerformance.csv'), preprocessing_for_bodyPreformance)
    # print("-" * 100)
    # decision_dependent_distance_based_knn_test(pd.read_csv(r'datasets/bodyPerformance.csv'), preprocessing_for_bodyPreformance, pre_trained_list)
    # print("-" * 100)
    # decision_independent_direct_knn_test(pd.read_csv(r'datasets/bodyPerformance.csv'), preprocessing_for_bodyPreformance, pre_trained_list)
    # print("-" * 100)
    # decision_independent_distance_based_knn_test(pd.read_csv(r'datasets/bodyPerformance.csv'), preprocessing_for_bodyPreformance, pre_trained_list)
    # print("-" * 100)
    # clustering_selection_test(pd.read_csv(r'datasets/bodyPerformance.csv'), preprocessing_for_bodyPreformance, pre_trained_list)
    # print("\n\n")
