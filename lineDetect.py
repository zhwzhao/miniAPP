import math
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt
import time


class LineDetect:
    def __init__(self, image, boundingbox):
        self.img = image
        self.canvas = np.zeros(image.shape[:2])
        self.box_min_y = 2000
        self.boxes = boundingbox
        # 存储[x, y]类型的数据，其中x,y为直线和x轴y轴的交点
        self.xy_focus = []
        self.max_k = 15     # 当斜率超过此值时，默认为直线为垂直的

    # 根据过滤后的点通过线性回归拟合出最后一条直线
    def getKBbyLinearRegression(self, filted_center):
        model = linear_model.LinearRegression()
        filted_center = np.array(filted_center)
        k = (max(filted_center[:, 1]) - min(filted_center[:, 1])) / (
                max(filted_center[:, 0]) - min(filted_center[:, 0]))
        if k > self.max_k:
            # 返回斜率，和一个点坐标
            return k, [np.mean(filted_center[:, 0]), np.mean(filted_center[:, 1])]
        model.fit(np.expand_dims(filted_center[:, 0], 1), np.expand_dims(filted_center[:, 1], 1))

        # plt.scatter(np.expand_dims(filted_center[:, 0], 1), np.expand_dims(filted_center[:, 1], 1))
        # x = np.linspace(1750, 1850, 100)
        # print('KB:', model.intercept_, model.coef_)
        # y = model.coef_[0] * x + model.intercept_[0]
        # plt.plot(x, y)
        # plt.show()
        # 返回斜率，截距
        return model.coef_[0][0], model.intercept_[0]

    # 过滤掉已经确定的三条直线附近的box，返回剩余box的中心点
    def removeotherbox(self):
        filter_box_center = []
        for box in self.boxes:
            x = (box[0] + box[2]) // 2
            y = (box[1] + box[3]) // 2
            add = True
            for kb in self.xy_focus:
                b, k = kb[1:]
                point1_dis = (k * box[0] + b - box[1]) / math.sqrt(1 + k * k)
                point2_dis = (k * box[2] + b - box[3]) / math.sqrt(1 + k * k)
                center_dis = (k * x + b - y) / math.sqrt(1 + k * k)
                if point1_dis * point2_dis < 0 or abs(center_dis) < 60:
                    add = False
            if add:
                filter_box_center.append([x, y])
        return filter_box_center

    def getResult(self):
        # 计算出boundingbox中心点绘制再canvas上
        for box in self.boxes:
            cv2.rectangle(self.img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thickness=1)
            x = (box[0] + box[2]) // 2
            y = (box[1] + box[3]) // 2
            self.box_min_y = min(self.box_min_y, y)
            cv2.circle(self.canvas, (x, y), 4, (255, 255, 255), thickness=4)


        self.canvas = self.canvas.astype(dtype=np.uint8)
        # 进行霍夫直线运算
        lines = cv2.HoughLines(self.canvas, 5, np.pi / 45, 4, )
        # 对检测到的每一条线段
        for line in lines:
            # 霍夫变换返回的是 r 和 theta 值
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            # 确定x0 和 y0
            x0 = a * rho
            y0 = b * rho
            # 认为构建（x1,y1）,(x2, y2)
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * a)
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * a)
            # 计算截距
            if x2 == x1:
                continue
            k = (y2 - y1) / (x2 - x1)
            # x_distance,y_distance分别表示在x，y轴上的交点
            y_distance = y1 - k * x1
            x_distance = (self.box_min_y - y_distance) / k
            if k > -0.3 or k < -1.5:
                continue

            self.xy_focus.sort()
            if len(self.xy_focus) == 3:
                break

            flag = True
            # 确定直线
            for i in range(len(self.xy_focus)):
                if (self.xy_focus[i][0] - x_distance) * (self.xy_focus[i][1] - y_distance) < 0:
                    flag = False
                if abs(self.xy_focus[i][0] - x_distance) < 50 or abs(self.xy_focus[i][1] - y_distance) < 50:
                    flag = False
            if flag:
                self.xy_focus.append([x_distance, y_distance, k])
                cv2.line(self.canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        center = self.removeotherbox()
        for cc in center:
            cv2.circle(self.img, (cc[0], cc[1]), 4, (0, 0, 255), thickness=4)
        K, B = self.getKBbyLinearRegression(center)
        # print(K, B)
        if K > self.max_k:
            cv2.line(self.img, (int(B[0]), 0), (int(B[0]), 3000), (0, 0, 255), 2)
            cv2.line(self.canvas, (int(B[0]), 0), (int(B[0]), 3000), (0, 0, 255), 2)
        else:
            cv2.line(self.img, (0, int(B)), (int(-B / K), 0), (0, 0, 255), 2)
            cv2.line(self.canvas, (0, int(B)), (int(-B / K), 0), (0, 0, 255), 2)
        # self.canvas = cv2.resize(self.canvas, (1080, 720))
        # cv2.imshow('canvas', self.canvas)
        return self.img


def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        BndBoxLoc = [x1, y1, x2, y2]
        if ObjBndBoxSet.__contains__(ObjName):
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet


if __name__ == '__main__':
    imgpath = 'data/images'
    xmlpath = 'data/Annotations'
    for file in os.listdir(imgpath):
        time1 = time.time()
        # file = '22607.png'
        img_path = os.path.join(imgpath, file)
        xml_path = os.path.join(xmlpath, file.split('.')[0] + '.xml')
        img = cv2.imread(img_path)
        boxes = GetAnnotBoxLoc(xml_path)['person']
        Line = LineDetect(img, boxes)
        img = Line.getResult()
        img = cv2.resize(img, (1080, 720))
        cv2.imshow('img', img)
        # cv2.imwrite('data/save/'+file, img)
        cv2.waitKey(1000)
        print('finish:', file)
        cv2.destroyAllWindows()
