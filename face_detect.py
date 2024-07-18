import cv2
alg="Haarcascade_frontalface_default.xml"
h_c=cv2.CascadeClassifier(alg)
cam=cv2.VideoCapture(0)
while True:
    _,img=cam.read()
    grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=h_c.detectMultiScale(grayimg,1.3,4)
    for (x,y,w,h)in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,250),7)
    cv2.imshow("FaceDetection",img)
    key=cv2.waitKey(10)
    if key==27:
        break

cam.release()
cv2.destroyAllWindows()
