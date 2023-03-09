import cv2

# img = cv2.imread ("C:\\Users\\agraw\\OneDrive\\Desktop\\BATTLEFIELD\\Python\\Project\\pimage.png",1)
# print(img)
# print(type(img))
# print(img.shape)
# img2 = cv2.imread ("C:\\Users\\agraw\\OneDrive\\Desktop\\BATTLEFIELD\\Python\\Project\\pimage.png",0)
# print(img2)
# print(type(img2))
# print(img2.shape)
# cv2.imshow("legend",img)
# cv2.waitKey()
# cv2.destroyAllWindows
# resized = cv2.resize(img, (int(img.shape[1]*2),int(img.shape[0]*2)))
# cv2.imshow("legend",resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img3 = cv2.imread("Sachin.jpg")
gray_img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img3,scaleFactor = 1.05, minNeighbors = 5)
for x,y,w,h in faces :
    img3 = cv2.rectangle(img3 , (x,y),(x+w,y+h),(0,255,0),3)
resized1 = cv2.resize(img3, (int(img3.shape[1]),int(img3.shape[0])))
cv2.imshow("gray",resized1)
cv2.waitKey(0)
cv2.destroyAllWindows    

