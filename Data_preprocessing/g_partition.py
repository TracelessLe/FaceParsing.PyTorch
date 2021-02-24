import os
import shutil
import pandas as pd
from shutil import copyfile
from utils import make_folder

#### source data path
s_label = 'CelebAMaskHQ-mask'
s_img = 'CelebA-HQ-img'
#### destination training data path
d_train_label = 'train_label'
d_train_img = 'train_img'
#### destination testing data path
d_test_label = 'test_label'
d_test_img = 'test_img'
#### val data path
d_val_label = 'val_label'
d_val_img = 'val_img'

#### make folder
make_folder(d_train_label)
make_folder(d_train_img)
make_folder(d_test_label)
make_folder(d_test_img)
make_folder(d_val_label)
make_folder(d_val_img)

#### calculate data counts in destination folder
train_count = 0
test_count = 0
val_count = 0

image_list = pd.read_csv('CelebA-HQ-to-CelebA-mapping.txt', delim_whitespace=True, header=0)
f_train = open('train_list.txt', 'w')
f_val = open('val_list.txt', 'w')
f_test = open('test_list.txt', 'w')

f_train_list, f_val_list, f_test_list = [], [], [] # store

for idx, x in enumerate(image_list.iloc[:, 1]):
    print ('processing: ', idx, x)
    if x >= 162771 and x < 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_val_label, str(val_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_val_img, str(val_count)+'.jpg'))        
        val_count += 1
        f_val_list.append(idx)
    elif x >= 182638:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_test_label, str(test_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_test_img, str(test_count)+'.jpg'))
        test_count += 1 
        f_test_list.append(idx)
    else:
        copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_train_label, str(train_count)+'.png'))
        copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_train_img, str(train_count)+'.jpg'))
        train_count += 1
        f_train_list.append(idx)

print ('Total Num: ', train_count + test_count + val_count)


#### write in and close the file
for i in f_train_list:
    f_train.write(str(i))
    f_train.write('\n')

for i in f_val_list:
    f_val.write(str(i))
    f_val.write('\n')

for i in f_test_list:
    f_test.write(str(i))
    f_test.write('\n')

f_train.close()
f_val.close()
f_test.close()
