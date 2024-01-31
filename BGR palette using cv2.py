import cv2
import numpy as np

def emptyFunc(x):
  return

img1 = np.zeros((512,512,3), np.uint8)

winName = ('OpenCV BGR Color Palette')
cv2.namedWindow(winName)

cv2.createTrackbar('B', winName, 0, 255, emptyFunc)
cv2.createTrackbar('G', winName, 0, 255, emptyFunc)
cv2.createTrackbar('R', winName, 0, 255, emptyFunc)

while(True):
  blue = cv2.getTrackbarPos('B', winName)
  green = cv2.getTrackbarPos('G', winName)
  red = cv2.getTrackbarPos('R', winName)

  img1[:] = [blue, green, red]
  cv2.imshow(winName, img1)
  if cv2.waitKey(1) == 27:
    break
cv2.destroyAllWindows()

