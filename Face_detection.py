import cv2

modelPath = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(modelPath)

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Camera not accessible")
    exit()

while True:
    status, image = cam.read()
    if not status:
        break

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray_image, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

    cv2.imshow("Face Detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
