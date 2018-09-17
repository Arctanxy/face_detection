import pandas as pd
import os
import cv2
from tqdm import tqdm
import pickle
from cut_face import cut_face

class DataSet:
    def __init__(self):
        self.dir = 'H:/face_detection/data/source_data'
        self.classes = self.get_class()
        self.data,self.labels = self.get_data_labels()

    def get_class(self):
        classes = [dirs for root,dirs,files in os.walk(self.dir) if dirs != []][0]
        return classes

    def get_data_labels(self,start=0,end=10000):
        if not os.path.exists('H:/face_detection/data/processed_data/0.jpg') and not os.path.exists('H:/face_detection/data/labels.pkl'):
            images = []
            labels = []
            for c in tqdm(self.classes):
                path = self.dir + '/' + c
                for root,dirs,file_name in os.walk(path):
                    for file in file_name:
                        img_path = path + '/' + file
                        cutted_face = cut_face(img_path)
                        if cutted_face is None:
                            cutted_face = cv2.imread(img_path) # 图片本身都是250×250的
                            cutted_face = cv2.resize(cutted_face, (150, 150), interpolation=cv2.INTER_CUBIC)
                            images.append(cutted_face)
                        else:
                            cutted_face = cv2.resize(cutted_face,(150,150),interpolation=cv2.INTER_CUBIC)
                            images.append(cutted_face)
                        labels.append(c)
            images = images[start:end]
            labels = labels[start:end]
            for i,img in tqdm(enumerate(images)):
                cv2.imwrite('H:/face_detection/data/processed_data/%d.jpg' % i,img)
            labels_out = open('H:/face_detection/data/labels.pkl','wb')
            pickle.dump(labels,labels_out)
            labels_out.close()
        else:
            labels_input = open('H:/face_detection/data/labels.pkl','rb')
            labels = pickle.load(labels_input)
            labels_input.close()
            images_path = ['H:/face_detection/data/processed_data/%d.jpg' % i for i in range(start,end)]
            images = []
            for img in tqdm(images_path):
                images.append(cv2.imread(img))
            labels = labels[start:end]
        return images,labels

if __name__ == "__main__":
    d = DataSet()
    print(d.labels)