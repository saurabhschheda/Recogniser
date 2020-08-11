import cv2
import pandas as pd
import pickle

class Predictor:
    people = [person[0] for person in pd.read_excel("../data/People.csv").values]
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        file = open("../data/classifier.pkl", "rb")
        self.model = pickle.load(file)
 
    def predict(self, image_paths):
        for image_path in image_paths:
            image = cv2.imread(image_path)
            faces_list = self.preprocessor.detect_faces(image)
            encoded_list = self.preprocessor.encode_detected_faces(faces_list)
            if encoded_list is not None:
                for encoded_image in encoded_list:
                    predictions_proba = self.model.predict_proba(encoded_image)[0].tolist()
                    # predictions = self.model.predict(encoding)
                    highest_confidence = max(predictions_proba)
                    person_name = "Unknown Person"
                    if highest_confidence > 0.6:
                        person_name = self.people[predictions_proba.index(highest_confidence)]
                    return person_name