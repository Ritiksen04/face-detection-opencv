import cv2
import cv2.data
modelPath = cv2.data.haarcascades + '/haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(modelPath)

cam = cv2.VideoCapture(0)
while True:
    status, image = cam.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_image, 1.3, 5)
    for face in faces:
        x = face[0]
        y = face[1]
        a = face[2]
        r = face[3]
        image = cv2.rectangle(image, (x,y) ,(x+a,y+r),(255,255,0),2)

        #image = cv2.circle(image, )
        cv2.imshow("faces", image)
        if cv2.waitKey(1)==ord('q'):
            break