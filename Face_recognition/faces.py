import cv2

imagePath = "images/stuff/people.jpg"
cascPath = "cascades/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.06,
    minNeighbors=5,
    minSize=(30, 30)
    # flags = cv2.CV.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, 'person', (x, y-5), 3, 0.7, (0,0,0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)