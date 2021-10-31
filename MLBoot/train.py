import sys
import json
import numpy as np
import pandas as pd
import spacy
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

def read_data_file(path_to_file):
    with open(path_to_file, "r") as data_file:
        dialogues = json.load(data_file)
    return dialogues

def train_nn_and_save_results(X_train, y_train, X_test, test_argument_id_list, nn_best_params_dict):
    clf = MLPClassifier(**nn_best_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_neural_networks.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)


def train_perceptron_and_save_results(X_train, y_train, X_test, test_argument_id_list, perceptron_best_params_dict):
    clf = Perceptron(**perceptron_best_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_perceptron.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)


def train_kNN_and_save_results(X_train, y_train, X_test, test_argument_id_list, kNN_best_params_dict):
    clf = KNeighborsClassifier(**kNN_best_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_kNN.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)


def train_svm_and_save_results(X_train, y_train, X_test, test_argument_id_list, svm_best_params_dict):
    clf = SVC(**svm_best_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_svm.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)


def train_lr_and_save_results(X_train, y_train, X_test, test_argument_id_list, lr_best_params_dict):
    clf = LogisticRegression(**lr_best_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_lr.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)


def train_random_forest_and_save_results(X_train, y_train, X_test, test_argument_id_list, random_forest_params_dict):
    clf = RandomForestClassifier(**random_forest_params_dict)
    clf.fit(X_train, y_train)
    prediction_list = clf.predict(X_test)
    prediction_list = [int(pred) for pred in prediction_list]

    prediction_dict = {}
    for index in range(len(prediction_list)):
        prediction_dict[test_argument_id_list[index]] = prediction_list[index]

    with open('predictions_random_forest.json', 'w') as pred_file:
        json.dump(prediction_dict, pred_file, indent=4)

nlp = spacy.load("en_core_web_sm")

train_dialogues = read_data_file('ca21-assignment-3/train-data-prepared.json')
train_merged_cleaned_dialogues = preprocess_text_data(merge_dialogue_texts(train_dialogues))

X_train = extract_features(train_merged_cleaned_dialogues, train_dialogues)
y_train = pd.Series(data=np.array([dialogue['label'] for dialogue in train_dialogues]))

test_file_path = sys.argv[1]

test_dialogues = read_data_file(test_file_path)
test_merged_cleaned_dialogues = preprocess_text_data(merge_dialogue_texts(test_dialogues))

X_test = extract_features(test_merged_cleaned_dialogues, test_dialogues)
y_test = pd.Series(data=np.array([dialogue['label'] for dialogue in test_dialogues]))

train_tfidf_features_df, test_tfidf_features_df = extract_tfidf_features(train_merged_cleaned_dialogues,
                                                                         test_merged_cleaned_dialogues, 150)

X_train = pd.concat([X_train, train_tfidf_features_df], axis=1)
X_test = pd.concat([X_test, test_tfidf_features_df], axis=1)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

