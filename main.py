#
import decisiontree as dt
import randomforest as rf
import csv

def main(csvfile):
    with open(csvfile, 'rb') as file:
        reader = csv.reader(file)
        dataset = list(reader)

    all_data = dataset[1:]
    featrues = dataset[0][:-1]

    #bags = rf.divbootstrap(all_data, 2)
    #print bags

    split = int (0.8*len(all_data))
    trainingset = all_data[:split]
    testset = all_data[split:]
    #sub_data, feat = rf.subfeatures(trainingset)
    #print feat
    test_value = []
    test_label = []
    for test in testset:
        test_value.append(test[:-1])
        test_label.append(test[-1])
    tennisTrees = rf.buildtrees(trainingset,featrues,20)
    #prec = random_predict(tennisTrees, test_value,tennisfeatrues)
    #print prec
    a = rf.random_accuracy(tennisTrees,test_value,featrues,test_label)
    print 'The accuracy is ', 100*a, '% .'


print "random forest: "
print "For tnnis dataset:"
main('datafile/tennis.csv')
print "For banks dataset:"
main('datafile/banks.csv')
print "For politics dataset:"
main('datafile/politics.csv')
