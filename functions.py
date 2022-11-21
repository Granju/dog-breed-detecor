import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tqdm import tqdm
from Model.model import load_model
from extract_bottleneck_features import extract_Resnet50

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def predict_breed_Resnet50(path):
    Resnet50_model = load_model()
    #Extract bottleneck features
    bottleneck_features = extract_Resnet50(path_to_tensor(path))
    #Get predictions vector
    predictions_vector = Resnet50_model.predict(bottleneck_features)
    #Return he name of the predicted dog breed
    return dog_names[np.argmax(predictions_vector)]

def dogbreedpredictor(path):

    if dog_detector(path):
        pred = predict_breed_Resnet50(path)
        print('This good boy looks like a {}.'.format(pred))

    elif face_detector(path):
        pred = predict_breed_Resnet50(path)
        print('This human looks like a {}.'.format(pred))

    else:
        print('No dog or human faces have been detected. Please make sure to' +
        ' upload a picture with a clearly visible human or dog face.')

print(extract_Resnet50(path_to_tensor('images/American_water_spaniel_00648.jpg')).shape)
