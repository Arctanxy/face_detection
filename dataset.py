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
        # 根据文件夹名称建立标签
        classes = [dirs for root,dirs,files in os.walk(self.dir) if dirs != []][0]
        return classes

    def get_data_labels(self,start=0,end=10000):
        # 获取数据和标签
        # 如果头像数据和标签数不存在，则重新使用opencv截取头像
        if not os.path.exists('H:/face_detection/data/processed_data/0.jpg') and not os.path.exists('H:/face_detection/data/labels.pkl'):
            images = []
            labels = []
            for c in tqdm(self.classes):
                path = self.dir + '/' + c
                for root,dirs,file_name in os.walk(path):
                    for file in file_name:
                        img_path = path + '/' + file
                        # 使用opencv截取头像
                        cutted_face = cut_face(img_path)
                        # 如果截取失败，则使用原图
                        if cutted_face is None:
                            cutted_face = cv2.imread(img_path)
                            cutted_face = cv2.resize(cutted_face, (150, 150), interpolation=cv2.INTER_CUBIC)
                            images.append(cutted_face)
                        else:
                            cutted_face = cv2.resize(cutted_face,(150,150),interpolation=cv2.INTER_CUBIC)
                            images.append(cutted_face)
                        labels.append(c)
            images = images[start:end]
            labels = labels[start:end]
            # 保存截取后的图像
            for i,img in tqdm(enumerate(images)):
                cv2.imwrite('H:/face_detection/data/processed_data/%d.jpg' % i,img)
            labels_out = open('H:/face_detection/data/labels.pkl','wb')
            pickle.dump(labels,labels_out)
            labels_out.close()
        # 如果截取后的头像文件存在
        else:
            labels_input = open('H:/face_detection/data/labels.pkl','rb')
            labels = pickle.load(labels_input)
            labels_input.close()
            # 返回path，而不是image
            images_path = ['H:/face_detection/data/processed_data/%d.jpg' % i for i in range(start,end)]
            labels = labels[start:end]
        return images_path,labels

if __name__ == "__main__":
    d = DataSet()
    print(d.labels)