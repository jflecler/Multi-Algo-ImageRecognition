import os, sys, time

from RecognitionImage import *
from Train import *

def loop_image_SetFolder(function, train):
	error, finded = 0, 0
	path = './data/trainingSet/trainingSet/'
	print("Dataset : "+path)
	directories = os.listdir(path)
	for file_ in directories:
		number = int(file_)
		print("Testing in folder : "+file_)
		path = './data/trainingSet/trainingSet/'+file_+'/'
		images = os.listdir(path)
		for img in images:
			result = function(path+img, train)
			if result == number:
				finded += 1
			else:
				error += 1
	return error, finded

def loop_image_SampleFolder(function, train):
	error, finded = 0, 0
	path = './data/trainingSample/trainingSample/'
	print("Dataset : "+path)
	directories = os.listdir(path)
	for file_ in directories:
		number = int(file_)
		print("Testing in folder : "+file_)
		path = './data/trainingSample/trainingSample/'+file_+'/'
		images = os.listdir(path)
		for img in images:
			result = function(path+img, train)
			if result == number:
				finded += 1
			else:
				error += 1
	return error, finded

def test_HDRTD(fonction_test):
	print("\n============== DECISION TREE ALGO ==============")
	print("Launch Training...")
	begin = time.time()
	clf = DT_train_model()
	end = time.time()
	print("Training is over, time for training : "+str(end-begin)+"s")
	print("Launch Loop Image Testing")
	begin = time.time()
	error, finded = fonction_test(RecognitionImage, clf)
	end = time.time()
	print("Loop Image is over, time for testing : "+str(end-begin)+"s")
	print("Result for the Decision Tree Algorithm : "+str(finded)+" nombres trouvé et "+str(error)+" incorrect.\n")

def test_HDRPer(fonction_test):
	print("\n============== PERCEPTRON LINEAR ALGO ==============")
	print("Launch Training...")
	begin = time.time()
	clf = PER_lin_train_model()
	end = time.time()
	print("Training is over, time for training : "+str(end-begin)+"s")
	print("Launch Loop Image Testing")
	begin = time.time()
	error, finded = fonction_test(RecognitionImage, clf)
	end = time.time()
	print("Loop Image is over, time for testing : "+str(end-begin)+"s")
	print("Result for the Perceptron Algorithm : "+str(finded)+" nombres trouvé et "+str(error)+" incorrect.\n")

def test_HDRPer_NN(fonction_test):
	print("\n============== PERCEPTRON NEURAL-NETWORK ALGO ==============")
	print("Launch Training...")
	begin = time.time()
	clf = PER_nn_train_model()
	end = time.time()
	print("Training is over, time for training : "+str(end-begin)+"s")
	print("Launch Loop Image Testing")
	begin = time.time()
	error, finded = fonction_test(RecognitionImage, clf)
	end = time.time()
	print("Loop Image is over, time for testing : "+str(end-begin)+"s")
	print("Result for the Perceptron Algorithm : "+str(finded)+" nombres trouvé et "+str(error)+" incorrect.\n")


if __name__ == '__main__':
	test_HDRTD(loop_image_SampleFolder)
	test_HDRPer(loop_image_SampleFolder)
	test_HDRPer_NN(loop_image_SampleFolder)
	test_HDRTD(loop_image_SetFolder)
	test_HDRPer(loop_image_SetFolder)
	test_HDRPer_NN(loop_image_SetFolder)