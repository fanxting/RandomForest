import decisiontree as dt
import csv

def main_dt(csvfile):
    with open(csvfile, 'rb') as file:
        reader = csv.reader(file)
        dataset = list(reader)
    total_data = dataset[1:]
    features = dataset[0][:-1]

    split = int (0.8*len(total_data))            #split the training set and test set
    trainingset = total_data[:split]
    testset = total_data[split:]

    test_value = []
    test_label = []
    for test in testset:
        test_value.append(test[:-1])
        test_label.append(test[-1])

    tree = dt.createTree(trainingset, features)
    #print tree

    features = dataset[0][:-1]
    #print features
    #createPlot(tree)
    taccuracy = dt.accuracy(tree,test_value,test_label,features)

    print 'The accuracy is', 100*taccuracy, '% .'


print "ID3:"
print "For tnnis dataset:"
main_dt('datafile/tennis.csv')
print "For banks dataset:"
main_dt('datafile/banks.csv')
print "For politics dataset:"
main_dt('datafile/politics.csv')