import xml.etree.ElementTree as ET
import os
import cv2
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from PIL import Image


def make_xml(boxes, size, image_name):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    # node_object_num = SubElement(node_root, 'object_num')
    # node_object_num.text = str(len(xmin_tuple))

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(size[0])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(size[1])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for cls in boxes.keys():
        for box in boxes[cls]:

            if cls == '2':
                continue
            elif cls == '0':
                tmp = '2'
            elif cls == '3':
                tmp = 0
            else:
                tmp = cls
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = str(tmp)
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(box[0])
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(box[1])
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(box[2])
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(box[3])
    xml = tostring(node_root, pretty_print=True)
    dom = parseString(xml)
    return dom


# get object annotation bndbox loc start
def GetAnnotBoxLoc(AnotPath):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    size = root.find('size')
    shape = [int(size.find('width').text), int(size.find('height').text)]
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text.split('.')[0])  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text.split('.')[0])  # -1
        x2 = int(BndBox.find('xmax').text.split('.')[0])  # -1
        y2 = int(BndBox.find('ymax').text.split('.')[0])  # -1
        BndBoxLoc = [x1, y1, x2, y2]
        if ObjBndBoxSet.__contains__(ObjName):
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet, shape


if __name__ == '__main__':
    path = r'C:\Users\zhao\Desktop\exam_data\220result'
    save_xml_path = r'C:\Users\zhao\Desktop\exam_data\tmp'
    for file in os.listdir(path):
        # file = '00216.xml'
        xml_path = os.path.join(path, file)

        boxes, size = GetAnnotBoxLoc(xml_path)
        print(boxes.keys())
        dom = make_xml(boxes, size, file.replace('xml', 'jpg'))



        out_file = open(os.path.join(save_xml_path, file), 'w', encoding='UTF-8')
        dom.writexml(out_file, indent='', addindent='\t', newl='\n', encoding='UTF-8')
        out_file.close()