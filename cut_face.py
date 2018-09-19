import cv2

def cut_face(path):
    # 使用预训练好的特征提取文件来探测脸部范围
    face_cascade = cv2.CascadeClassifier(r'G:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

    img = cv2.imread(path)

    faces = face_cascade.detectMultiScale(img,1.3,5)

    for (x,y,w,h) in faces:
        roi_color = img[y:y+h,x:x+w]
        # 直接返回第一个找到的人脸
        return roi_color
