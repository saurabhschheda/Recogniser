import Preprocessor
import cv2
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

class Trainer:
    dataset = pd.read_excel("../data/Dataset.csv").values
    people = [person[0] for person in pd.read_excel("../data/People.csv").values]    

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def update_dataset(self, image_paths, name):
        encoded_images = self.preprocessor.get_training_set_encoding(image_paths)
        temp_dataset = self.dataset.flatten()

        try:
            label = self.people.index(name)
        except ValueError:
            label = len(self.people)
            self.people.append(name)
            people_df = pd.DataFrame(self.people)
            people_df.to_csv("../data/People.csv", index=False)

        for encoded_image in encoded_images:
            encoded_image = np.append(encoded_image, label)
            temp_dataset = np.append(temp_dataset, encoded_image, axis=0)

        
        temp_dataset.shape = (int(temp_dataset.shape[0]/129), 129)
        self.dataset = temp_dataset
        self.train_model()
        dataset_df = pd.DataFrame(temp_dataset)
        dataset_df.to_excel("../data/Dataset.csv", index=False)
    
    def train_model(self):
        data = self.dataset[:, :128]
        labels = self.dataset[:, -1]
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        # print("Training for {} classes.".format(nClasses))
        model = SVC(C=1, kernel='linear', probability=True)
        model.fit(data, labelsNum)
        fName = "../data/classifier.pkl"
        print("Saving classifier")
        with open(fName, 'wb') as f:
            #pickle.dump((le, model), f)
            pickle.dump(model, f)
