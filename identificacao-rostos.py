import cv2

cascadeFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imgOriginal = cv2.imread("selecao-italia.jpeg")
imagem = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

faces = cascadeFace.detectMultiScale(
    imagem, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(imgOriginal, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(len(faces))

cv2.imshow("Resultado", imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()
