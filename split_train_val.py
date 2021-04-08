import os
import numpy as np
import random

path = r'./labels'


ll = []
test = []
for txt in os.listdir(path):
    if txt[0] == '1':
        test.append(txt.replace('txt', 'jpg'))
    else:
        ll.append(txt.replace('txt', 'jpg'))

random.shuffle(ll)
random.shuffle(test)
print(len(ll))

image_path = r'C:\Users\zhao\Desktop\exam_data\images'

f = open('train.txt', 'w')
for ff in ll:
    ss = os.path.join(image_path, ff)
    f.write(ss+'\n')
f.close()

f = open('test.txt', 'w')
for ff in test:
    ss = os.path.join(image_path, ff)
    f.write(ss + '\n')
f.close()