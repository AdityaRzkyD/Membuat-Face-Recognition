#file ini digunakan untuk deteksi wajah
#dan disimpan dalam file csv menggunakan npwriter

import cv2
import numpy as np
import npwriter

file_name=input("Enter the name of the person : ")

skip=0
face_data=[]

#digunakan untuk mengakses web-cam
#kemudian menangkap frame
cap = cv2.VideoCapture(0)

#class ini berfungsi untuk mendeteksi wajah
#berdasarkan dataset haarcascode_frontalface_default.xml
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

f_list = []

while True:

    ret, frame = cap.read()

    #konversi gambar ke format grayscale
    #untuk mempermudah deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #mendeteksi wajah dan koordinatnya
    faces = classifier.detectMultiScale(gray, 1.5, 5)

    #digunakan untuk mendeteksi wajah yang paling dekat
    #dengan web-cam pada posisi pertama
    faces = sorted(faces, key = lambda x: x[2]*x[3], reverse = True)

    #hanya wajah yang terdeteksi pertama kali
    #yang digunakan
    faces = faces[:1]

    for face in faces[-1:]:
        x, y, w, h=face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255,255), 2)

        offset=10
        face_section=frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
        
        cv2.imshow("Video Frame", frame)
        cv2.imshow("Face Section", face_section)

    if not ret:
            continue

    key = cv2.waitKey(1)

    #eksekusi program akan berhenti jika this will break the execution of the program
    #menekan tombol "q" dan menekan frame + menekan tombol "c"

    if key & 0xFF == ord('q'):
         break
    elif key & 0xFF == ord('c'):
        if len(faces) == 1:
            gray_face = cv2.cvtColor(face_section, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (100, 100))
            print(len(f_list), type(gray_face), gray_face.shape)

            #menambahkan koordinat wajah pada f_list
            f_list.append(gray_face.reshape(-1))
        else:
            print("face not found")

        #menyimpan data hasil deteksi wajah
        #sebanyak 10 untuk meningkatkan akurasi
        if len(f_list) == 10:
             break
        
#menyimpan data menggunakan npwriter
npwriter.write(file_name, np.array(f_list))

cap.release()
cv2.destroyAllWindows()