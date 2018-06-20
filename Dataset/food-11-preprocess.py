import numpy as np
import pickle
import os, glob
import cv2
import augmentation as aug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

train_path = 'Food-11/training/'
validate_path = 'Food-11/validation/'

n_train_data = len(glob.glob(os.path.join(train_path+'*')))
n_validate_data = len(glob.glob(os.path.join(validate_path+'*')))

min_ = 207, 220, 1.0

train_data_pkl = 'food-11_train.pkl'
validate_data_pkl = 'food-11_validate.pkl'


IMAGE_SIZE = aug.IMAGE_SIZE

  
def resize(frame):
	new_img = abs(np.array(np.random.normal(0.3, 0.3, (IMAGE_SIZE, IMAGE_SIZE, 3)), dtype=np.float32))


	h, w, _ = frame.shape
	if h > w:
		ratio = IMAGE_SIZE/h
		border = int((IMAGE_SIZE - int(w*ratio))/2)

		try: new_img[:,border:border+int(w*ratio),:] = cv2.resize(frame, (int(w*ratio), 180))/255
		except: 
			print(h, w, ratio)
			print(int(w*ratio), int(h*ratio))
			print(border, border+int(w*ratio))
			print(cv2.resize(frame, (int(w*ratio), int(h*ratio))).shape)

	else:
		ratio = IMAGE_SIZE/w
		border = int((IMAGE_SIZE - int(h*ratio))/2)
		try: new_img[border:border+int(h*ratio),:,:] = cv2.resize(frame, (180, int(h*ratio)))/255
		except: 
			print(h, w, ratio)
			print(int(w*ratio), int(h*ratio))
			print(border, border+int(h*ratio))
			print(cv2.resize(frame, (int(w*ratio), int(h*ratio))).shape)


	# cv2.imshow('',new_img)
	# cv2.waitKey(1)

	return new_img

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]


n_ = 0

for img_name in np.sort(glob.glob(os.path.join(train_path+'*'))):
	n_+=1
	print('\r%d/%d (%5.3f%%)'%(n_, n_train_data, n_/(n_train_data)*100), end='')
	n_class = int(img_name.split('/')[-1].split('_')[0])
	
	#print('\n'+img_name)
	a = cv2.imread(img_name)
	a = resize(a)
	
	
	X_train.append(a)
	Y_train.append(n_class)
	
	#if len(X_train)==100: break;


print(' Train Data Size:', len(X_train))


n_ = 0
for img_name in np.sort(glob.glob(os.path.join(validate_path+'*'))):
	n_+=1
	print('\r%d/%d (%5.3f%%)'%(n_, n_validate_data, n_/(n_validate_data)*100), end='')
	n_class = int(img_name.split('/')[-1].split('_')[0])
	
	a = cv2.imread(img_name)
	a = resize(a)
	
	X_test.append(a)
	Y_test.append(n_class)

	#if len(X_test)==100: break;



print(' Validation Data Size:', len(X_test))

ALL_DATA = {}
ALL_DATA['X_train'] = X_train
ALL_DATA['Y_train'] = Y_train
ALL_DATA['X_test'] = X_test
ALL_DATA['Y_test'] = Y_test

#save = ALL_Data(X_train, Y_train, X_test, Y_test)


with open('food-11_180.pkl', 'wb') as pkl: 
	pickle.dump(ALL_DATA, pkl)

