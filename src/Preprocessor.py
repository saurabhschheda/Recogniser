import cv2
import dlib
import openface
import numpy as np

class Preprocessor:
    face_aligner = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
    face_detector = dlib.get_frontal_face_detector()
    face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    assign_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    num_jitters = 1
    size = 300
    rect = dlib.rectangle(0, 0, size, size)

    def detect_faces(self, image):
        # Detects the faces and returns a rectangle of coordinates [(left, top) (right, bottom)]
        detected_faces = self.face_detector(image, 1)
        # To Obtain the rectangle of coordinates as a list
        coordinates = list()
        for face_array in detected_faces:
            # Rectangle is made into a string to be read
            rectangle = str(face_array)
            face_coordinates = list()
            number_str = ""
            for i in rectangle:
                if i.isdigit():
                    number_str += i
                elif i == ',' or i == ')':
                    face_coordinates += [int(number_str)]
                    number_str = ""
            coordinates.append(face_coordinates)
        # To Obtain a 2D array of gray scale faces in a list
        faces = list()
        for coordinate in coordinates:
            face = image[coordinate[1]:coordinate[3]+1, coordinate[0]:coordinate[2]+1]
            faces.append(face)
        return faces
        # return self.encode_detected_faces(faces)


    def encode_detected_faces(self, faces):
        measurements = list()
        flag = 0
        for face in faces:
            try:
                aligned_face = self.face_aligner.align(self.size, face, landmarkIndices=openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
                shape = self.assign_landmarks(aligned_face, self.rect)
                measurements.append(np.array(self.face_encoder.compute_face_descriptor(aligned_face, shape, self.num_jitters)))
                flag = 1
            except TypeError:
                pass
        if flag == 1:
            return measurements
        else:
            return None


    def get_training_set_encoding(self, image_paths):
        encoded_images = list()
        for image_path in image_paths:
            image = cv2.imread(image_path)
            # Check if a list containing numpy array is passed
            encoded = self.encode_detected_faces([image])
            if encoded is not None:
                encoded_images.append(encoded[0])
        return encoded_images
        
