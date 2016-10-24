# -*- coding: utf-8 -*-

import decisiontree as dt

import random
import operator
import math

def subfeatures(dataset):
    n_features = len(dataset[0])-1
    num_sub_features = int (math.sqrt(n_features))  # the number of sub_features
#    print num_sub_features
    feature_data = random.sample(xrange(n_features),num_sub_features)    #select the sub_features randomly
    sub_features = sorted(feature_data)   
#    print feature_sort
    sub_dataset=[]       
    for data in dataset:
        sub_data = []
        for i in sub_features:
            sub_data.append(data[i])
        sub_dataset.append(sub_data)
#    print sub_dataset
    return sub_dataset, sub_features     

def divbootstrap(dataset, ensemblesize):
    num = len(dataset)
    bags = []
    for i in range(ensemblesize):
        ensemble = []
        for j in range(num):
            ensemble.append(random.sample(dataset,1)[0]) 
#        print ensemble   
        bags.append(random.sample(ensemble,num))
    return bags          

##trainset: training set
#features: features list
#n: ensemble size
def buildtrees(trainset, features,n):
    trees = []
    bags  = divbootstrap(trainset, n)
    features_in = []
    for ensemble in bags:
        for fea in features:
            features_in.append(fea)
        tree = dt.createTree(ensemble, features_in)
        trees.append(tree)        
    return trees    

#classify test set
def random_predict(treesset, testdata, features, label):
    predicts = []
    for tree in treesset:
        predict = dt.predict(tree, features, testdata, label)
        predicts.append(predict)
    return majorvoting(predicts)

#major vote
def majorvoting(predict_result):
    classes = {}
    for vote in predict_result:
        if vote not in classes.keys():
            classes[vote] = 0
        classes[vote] += 1
        sortedclass = sorted(classes.iteritems(), key = operator.itemgetter(1), reverse = True) #sort the lalels
        return sortedclass[0][0]
                                                                                
#calculate the accuracy
def random_accuracy(treesset, test_value, features,testclass):
    predicts = []
    for testdata in test_value:
        prec = random_predict(treesset, testdata,features, testclass[0])
        predicts.append(prec)       #get the results of predict
    num = len(testclass)
    num_correct = 0
    for x in xrange(num):
        if predicts[x] == testclass[x]:
            num_correct = num_correct + 1   #get the num of correct result
    accuracy = round(float(num_correct)/num,4)    #calculate the accuracy
    return accuracy           
                        
