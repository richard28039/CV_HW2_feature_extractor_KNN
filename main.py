import os
import argparse
import cv2

from tqdm.auto import tqdm
from dataset import *
from feature_extraction.feature_extractor import * 
from feature_extraction.knn import * 


def makedir(path:os.PathLike)->os.PathLike:
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    except Exception as e:
        print(f"Error creating directory: {e}")

def extract_img(dataset:Dataset, save_dir:os.PathLike, feature, train_or_test):
    save_path = os.path.join(save_dir,opt.feature_extractor + '/' + train_or_test)
    if train_or_test == "train":
        if feature == "Gabor_Filters":
            for load in tqdm(dataset):
                # print(load[0].shape)
                cv2.imwrite(makedir(save_path + "/" + load[2]) + "/" + load[1], Gabor_filter(load[0]))
        if feature == "Local_Binary_Pattern":
            for load in tqdm(dataset):
                # print(load[0].shape)
                cv2.imwrite(makedir(save_path + "/" + load[2]) + "/" + load[1], lbp(load[0]))
        if feature == "Color_histogram":
            for load in tqdm(dataset):
                # print(load[0].shape)
                np.save(makedir(save_path + "/" + load[2]) + "/" + load[1], RGB_histogram(load[0]))
 
    if train_or_test == "test":
        makedir(save_path)
        if feature == "Gabor_Filters":
            for load in tqdm(dataset):
                # print(load[0].shape)
                cv2.imwrite(save_path + "/" + load[1] , Gabor_filter(load[0]))
        if feature == "Local_Binary_Pattern":
            for load in tqdm(dataset):
                # print(load[0].shape)
                cv2.imwrite(save_path + "/" + load[1] , lbp(load[0]))
        if feature == "Color_histogram":
            for load in tqdm(dataset):
                # print(load[0].shape)
                np.save(save_path + "/" + load[1] , RGB_histogram(load[0]))

def get_feature_img(feature_path:os.PathLike):
    X_train = [] 
    y_train = [] 
    X_test = []
    y_name = []

    feature_train = Dataset.get_train_dataset(feature_path + 'train/')
    for data in feature_train:
        X_train.append(data[0])
        X_test.append(data[2])

    feature_test = Dataset.get_test_dataset(feature_path + 'test/')
    for data in feature_test:
        y_train.append(data[0])
        y_name.append(data[1])

    return np.array(X_train), np.array(y_train), np.array(X_test),  np.array(y_name)

def get_feature_npy(feature_path:os.PathLike):
    X_train = [] 
    y_train = [] 
    X_test = []
    y_name = []

    feature_train = Dataset.get_train_npy(feature_path + 'train/')
    for data in feature_train:
        X_train.append(data[0])
        X_test.append(data[2])

    feature_test = Dataset.get_test_npy(feature_path + 'test/')
    for data in feature_test:
        y_train.append(data[0])
        y_name.append(data[1])

    return np.array(X_train), np.array(y_train), np.array(X_test),  np.array(y_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="",help = "get train data")
    parser.add_argument("--test",default="", help = "get test data")
    parser.add_argument("--feature_extractor", choices= ["Raw_image", "Color_histogram", "Local_Binary_Pattern", "Gabor_Filters"])
    parser.add_argument("--save_path", default="")
    parser.add_argument("--knn_predict", default="")
    opt = parser.parse_args()

    if len(opt.train):
        extract_img(Dataset.get_train_dataset(opt.train, get_feature = True), opt.save_path, opt.feature_extractor, "train")

    if len(opt.test):
        extract_img(Dataset.get_test_dataset(opt.test, get_feature = True), opt.save_path, opt.feature_extractor, "test")

    if len(opt.knn_predict):
        if opt.feature_extractor == "Gabor_Filters":

            X_train, y_train, X_test, y_nam = get_feature_img(opt.knn_predict)

            print(len(X_train), len(y_train), len(X_test), len(y_nam))

            label_mapping = {label: idx for idx, label in enumerate(set(X_test))}

            integer_labels_array = np.array([label_mapping[label] for label in X_test])

            result = knn_predict(X_train, y_train, integer_labels_array, 10)

            inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
            
            predicted_labels = np.array([inverse_label_mapping[idx] for idx in result])

            df = pd.DataFrame({"file" : y_nam ,"species" : predicted_labels})

            df.to_csv("Gabor_Filters_predict.csv", index=False)

        if opt.feature_extractor == "Local_Binary_Pattern":

            X_train, y_train, X_test, y_nam = get_feature_img(opt.knn_predict)

            print(len(X_train), len(y_train), len(X_test), len(y_nam))

            label_mapping = {label: idx for idx, label in enumerate(set(X_test))}

            integer_labels_array = np.array([label_mapping[label] for label in X_test])

            result = knn_predict(X_train, y_train, integer_labels_array, 10)

            inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
            
            predicted_labels = np.array([inverse_label_mapping[idx] for idx in result])

            df = pd.DataFrame({"file" : y_nam ,"species" : predicted_labels})

            df.to_csv("Local_Binary_Pattern_predict.csv", index=False)
        
        if opt.feature_extractor == "Raw_image":

            X_train, y_train, X_test, y_nam = get_feature_img(opt.knn_predict)

            print(len(X_train), len(y_train), len(X_test), len(y_nam))

            label_mapping = {label: idx for idx, label in enumerate(set(X_test))}

            integer_labels_array = np.array([label_mapping[label] for label in X_test])

            result = knn_predict(X_train, y_train, integer_labels_array, 10)

            inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
            
            predicted_labels = np.array([inverse_label_mapping[idx] for idx in result])

            df = pd.DataFrame({"file" : y_nam ,"species" : predicted_labels})

            df.to_csv("Raw_image.csv", index=False)

        if opt.feature_extractor == "Color_histogram":

            X_train, y_train, X_test, y_nam = get_feature_npy(opt.knn_predict)

            print(len(X_train), len(y_train), len(X_test), len(y_nam))

            label_mapping = {label: idx for idx, label in enumerate(set(X_test))}

            integer_labels_array = np.array([label_mapping[label] for label in X_test])

            result = knn_predict(X_train, y_train, integer_labels_array, 10)

            inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
            
            predicted_labels = np.array([inverse_label_mapping[idx] for idx in result])

            file_names = []

            for file_path in y_nam:
                base_name = os.path.basename(file_path)
                file_name_without_extension, _ = os.path.splitext(base_name)
                file_names.append(file_name_without_extension)

            df = pd.DataFrame({"file" : file_names ,"species" : predicted_labels})

            df.to_csv("Color_histogram.csv", index=False)
        