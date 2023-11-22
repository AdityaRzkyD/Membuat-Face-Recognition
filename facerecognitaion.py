#file ini digunakan untuk face recognition
#setelah data kita training
#menggunakan knn
import cv2
import numpy as np
import pandas as pd

from npwriter import f_name
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#membaca data
data = pd.read_csv(f_name).values

#membagi data
X, Y = data[:, 1:-1], data[:, -1]

print(X, Y)

#memanggil fungsi kNN dengan k = 5
model = KNeighborsClassifier(n_neighbors = 5)

#fdtraining dari model
model.fit(X, Y)

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

f_list =[]

while True:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(gray, 1.5, 5)

    X_test = []

    #testing data
    for face in faces:
        x, y, w, h = face
        im_face = gray[y:y + h, x:x + w]
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))

    if len(faces)>0:

        #hasil prediksi dari kNN
        response = model.predict(np.array(X_test))
        correct = np.count_nonzero(response)
        accuracy = (correct*100.0)/(response.size)

        for i, face in enumerate(faces):
            x, y, w, h = face

            #menggambar kotak pada wajah yang terdeteksi
            cv2.rectangle(frame, (x, y), ( x + w, y + h), (255, 0, 0), 3)

            #memberi label hasil prediksi
            cv2.putText(frame, response[i], (x-50, y-50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

            cv2.putText(frame, str(accuracy), (x+5, y+h-5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

        cv2.imshow("full", frame)

        key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()