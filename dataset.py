import os

import cv2
import pandas as pd
import numpy as np


def walkdir(root:os.PathLike)->list:
    ret = []
    for r,_,fs in os.walk(root):
        for f in fs:
            ret.append(os.path.join(r, f))
        return ret

class Dataset():
    def __init__(self, root :os.PathLike) -> None:
        pass

    @staticmethod
    def get_train_list(root:os.PathLike, csv_name = None):
        image_and_cls = []
        classes = [classes.name for classes in os.scandir(root)]
        for cl in classes:
            img_dir = os.path.join(root, cl)
            imgs_path = walkdir(img_dir)
            image_and_cls += [[img, cl] for img in imgs_path] 

        if csv_name:
            df = pd.DataFrame(image_and_cls)
            df.to_csv(csv_name, index=False)
        
        return np.array(image_and_cls)
    
    # 有點redundant但我懶得改
    @staticmethod
    def get_train_dataset(root:os.PathLike, get_feature = None):
        img_and_cls = []
        for data in Dataset.get_train_list(root):
            if get_feature == True:
                image  = cv2.cvtColor(cv2.imread(data[0]), cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image, (299, 299))
                img_and_cls += [[resized_image, os.path.split(data[0])[-1], data[1]]] 
            else:
                image  = cv2.imread(data[0])
                resized_image = cv2.resize(image, (299, 299))
                img_and_cls += [[resized_image, os.path.split(data[0])[-1], data[1]]] 

        return np.array(img_and_cls)

    @staticmethod
    def get_test_dataset(root:os.PathLike, get_feature = None):
        test_img = []
        for i in walkdir(root):
            if get_feature == True:
                image  = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image, (299, 299))
                test_img += [[resized_image, os.path.split(i)[-1]]]
            else:
                image  = cv2.imread(i)
                resized_image = cv2.resize(image, (299, 299))
                test_img += [[resized_image, os.path.split(i)[-1]]]

        return np.array(test_img)
    
    @staticmethod
    def get_train_npy(root:os.PathLike):
        img_and_cls = []
        for data in Dataset.get_train_list(root):
            vect = np.load(data[0])
            img_and_cls += [[vect, os.path.split(data[0])[-1], data[1]]] 

        return np.array(img_and_cls)

    @staticmethod
    def get_test_npy(root:os.PathLike):
        test_img = []
        for i in walkdir(root):
            vect = np.load(i)
            test_img += [[vect, os.path.split(i)[-1]]]

        return np.array(test_img)

if __name__ == "__main__":
    # Dataset.get_train_list("./plant-seedings-classification/train/", "train.csv")
    # print(Dataset.get_test_dataset("./plant-seedings-classification/test/"))
    print(Dataset.get_train_dataset("./plant-seedings-classification/train/").shape)

