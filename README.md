# RandomForest   

random forest classifier  
A random forest classifier written in python.     

There are three files here:  
decisiontree.py  
randomforest.py  
main.py  

How to run:    
1. run the random forest application:    
	just run the main.py file using the following comment:   
	   python main.py   
	or click run in IDE   

2. just run the decisiontree apllication:   
   a. read the following comments and do what it told you to run decision tree.   

***************************************************
# select the best branch to split, return the indext of the best feature  
def chooseBest_ToSplit(dataSet):  
    sub_data,feature_index = rf.subfeatures(dataSet)   #comment this line when just run decision tree  
    num_of_Features = len(dataSet[0]) - 1   #number of features  
    baseEntropy = cal_entropy(dataSet)  
    bestIG = 0.0;   
    bestFeature = -1            #intialize the index of the best feature  
#    for i in range(num_of_Features):     ##uncomment this code line when just run decision tree 
``
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

    ****************************************************
     
     b. then run main_dt.py file using the following comment:
	        python main_dt.py
	   or click run in IDE

 ``

decisiontree.py:    
	* implement ID3 algorithm including(main functions):  
	cal_entropy: calculate the entropy which can help to get the IG  
	chooseBest_ToSplit: find the best feature to split the node  
	creatTree: build a tree  
	prdict: for a given test data, to prodict it's class based on the tree  
	accuracy: calculate the accuracy  
 

randomforest.py:     
	implement 100% boostrap and randomly subspace:  
	divbootstrap: Build new 100% bootstrap samples by randomly selecting N examples from the training data with replacement  
	subfeatures: randomly choose m features and used it in the funciton chooseBest_ToSplit within the decisiontree.py  
	buildtrees: build a set of tree base on the bootstrap samples and sub features  
	random_predict: for a given test data, to prodict it's class based on the tree  
	majorvoting: Use majority voting to make predictions based on all T trees  
	random_accuracy:calculate the accuracy  
 
main.py  
	used to test. there are three datasets used: tennis.csv, banks.csv and politics.csv  
	used split to get the training set for every csv file 
	get the accuracy  
