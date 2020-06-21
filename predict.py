import os
import cv2
import glob
import numpy as np
from efficientnet.keras import EfficientNetB3
from keras.models import load_model
import csv

w, h = 224, 224
test_path = './test/test/'
model_path = './models/model_2.h5'
csv_path = './output_2.csv'

class Predict():
    def __init__(self):
        self.predict_list = []
        try:
            self.model = load_model(model_path)
        except ValueError:
            print('Model is not found!')

    def predict_data(self, f_path, image):
        image = (np.asarray(image)).reshape(-1, w, h, 3) / 255.
        result = self.model.predict(image)
        self.predict_list.append([f_path[-9:], result.argmax(axis=1)])
        
    def write_csv(self):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['image_id', 'label'])
            for x in self.predict_list:
                writer.writerow([x[0], x[1][0]])
            print('CSV writes finished!')

predict = Predict()
img_path = os.path.join(test_path, '*.jpg')
files = glob.glob(img_path)
print('Testing data are load.')

for f_path in files:
    img = cv2.imread(f_path)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    predict.predict_data(f_path, img)

predict.write_csv()