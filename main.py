import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


captured_image = cv2.VideoCapture(0)

while True:

    ret, img = captured_image.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        eye_color = img[y:y + h, x:x + w]
        face_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        eyes = eye_cascade.detectMultiScale(face_gray)

        for (x1, y1, w1, z1) in eyes:
            cv2.rectangle(eye_color, (x1, y1), (x1 + w1, y1 + z1), (0, 127, 255), 2)

    # To draw a rectangle in eyes
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

captured_image.release()
cv2.destroyAllWindows()