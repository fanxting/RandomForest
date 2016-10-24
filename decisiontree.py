# -*- coding: utf-8 -*-

import randomforest as rf

from math import log
import operator
#
#Calculte the Entropy
def cal_entropy(dataSet):
    num_Entries = len(dataSet)
    label_Counts = {}
    for dataVec in dataSet:
        curlabel = dataVec[-1]
        if curlabel not in label_Counts.keys():     #Create a dict for all probability types
            label_Counts[curlabel] = 0
        label_Counts[curlabel] += 1
    entropy = 0.0
    for key in label_Counts:
        p = float(label_Counts[key])/num_Entries
        entropy -= p * log(p,2)   #calculate entropy
    return entropy
          

##split the dataset based on the feature
#idx:index of feature
#value: the value of the feature
#return the dataset that split    
def split_DataSet(dataSet, idx, value):
    DataSet_after_split = []
    for data in dataSet:
        if data[idx] == value:
            data_reduced_feature_value = data[:idx]
            data_reduced_feature_value.extend(data[idx+1:])
            DataSet_after_split.append(data_reduced_feature_value)        #delete the value of the feature with index idx
    return DataSet_after_split

#select the best branch to split, return the indext of the best feature
def chooseBest_ToSplit(dataSet):
    sub_data,feature_index = rf.subfeatures(dataSet)   #comment this line when just run decision tree
    num_of_Features = len(dataSet[0]) - 1   #number of features
    baseEntropy = cal_entropy(dataSet)
    bestIG = 0.0; 
    bestFeature = -1            #intialize the index of the best feature
#    for i in range(num_of_Features):     ##uncomment this code line when just run decision tree
    for i in feature_index:     #traverse the i-th features  #comment this line when just run decision tree
        featureSet = set([data[i] for data in dataSet])   #the values of the i-th feature
        newEntropy= 0.0
        for value in featureSet:
            subDataSet = split_DataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * cal_entropy(subDataSet)   #entropy based on this feature
        curr_IG = baseEntropy - newEntropy       #get the current Information Gain
        if curr_IG > bestIG:                 # get the best Information Gain
            bestIG = curr_IG
            bestFeature = i
    return bestFeature
    
#Create a tree, save as a dict, then return this dict
def createTree(dataSet, features):
    classList = [data[-1] for data in dataSet]
    if classList.count(classList[0]) == len(classList):    #stop when all samples in this subset belongs to one class,return label--leaf node.
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityvotingclass(classList)       #Voting to get the majority of label in this subset if no extra features avaliable
    best_Feature = chooseBest_ToSplit(dataSet)
    bestFeature_Label = features[best_Feature]    # get the best splitting feature,add it to dictionary
    myTree = {bestFeature_Label:{}}
    del(features[best_Feature])
    featValues = [data[best_Feature] for data in dataSet]    #get all of the feature values
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subfeatures = features[:]
        myTree[bestFeature_Label][value] = createTree(split_DataSet(dataSet, best_Feature, value), subfeatures)
    return myTree
    
#majority voting to get the label of the leaf 
def majorityvotingclass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0;
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)  #Sort function in operator
    return sortedClassCount[0][0]
    

#Predict
def predict(inputTree, features, testdata_value, label):   #inputTree: tree; features: features, testdata: one testsample;lable:intial label
    # for test in testset:
    if type(inputTree).__name__ != 'dict':
        return inputTree
    first_feature_name = inputTree.keys()[0]    #get the name of first feature 
    secondDict = inputTree[first_feature_name]  
    feature_Index = features.index(first_feature_name)   #find the index of the first item that match the name of first feature
    class_Lables = label
  #  print class_Lables
    for key in secondDict.keys():
        if testdata_value[feature_Index] == key:
            if type(secondDict[key]).__name__ == 'dict':
                class_Lables = predict(secondDict[key], features, testdata_value,label)
            else: class_Lables = secondDict[key]
    return class_Lables
            
#Calculat the accuracy    
def accuracy(decisionTree,testdata,test_class_label, features):
    predict_test = []
    for data in testdata:
        predict_test.append(predict(decisionTree,features,data,test_class_label[0]))
    num = len(test_class_label)
    correct = 0
    for x in xrange(num):
        if predict_test[x] == test_class_label[x]:
            correct = correct + 1
    test_accuracy = round(float(correct)/num,4)
    return test_accuracy    

