import pickle
import numpy as np
'''

class ALL_Data():
	def __init__(self, X_train, Y_train, X_test, Y_test):
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_test = X_test
		self.Y_test = Y_test
'''
data=None

with open('food-11_180.pkl', 'rb') as f:
	data = pickle.load(f)

print(len(data['X_train']))
print(len(data['Y_train']))
print(len(data['X_test']))
print(len(data['Y_test']))
print(data['Y_test'])