import cv2
import numpy as np

def blur(img, k):
  h, w = img.shape[0:2]
  kw, kh = w//2, h//2
  if kw%2 == 0:
    kw -= 1
  if kh%2 == 0:
    kh -= 1
  img = cv2.GaussianBlur(img, ksize=(kw,kh), sigmaX=0)
  return img

def pixelate_img(img, blocks = 10):
  (h, w) = img.shape[0:2]
  xSteps = np.linspace(0, w, blocks + 1, dtype = 'int')
  ySteps = np.linspace(0, h, blocks + 1, dtype = 'int')
  for i in range(1, len(ySteps)):
    for j in range(1, len(xSteps)):
      startX = xSteps[j - 1]
      startY = ySteps[i - 1]
      endX = xSteps[j]
      endY = ySteps[i]

      roi = img[startY:endY, startX:endX]
      (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
      cv2.rectangle(img, (startX, startY), (endX, endY),
                (B, G, R), -1)
    # return the pixelated blurred image
    return img

face_cascade = cv2.CascadeClassifier(r'C:\Users\Nalini Ajwani\Downloads\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if frame is None:
        print("Error: Invalid frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()