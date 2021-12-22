import numpy as np
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#np.set_printoptions(threshold=sys.maxsize)
"""
Name: Rayhan Chowdhury


"""

#loading data sets
def load_dataset(dataset):
    """This method loads the data set, splits the testing and training between 80% and 20%, 
    and then returns the training data, training labels, testing data, testing labels"""
    pass

    raw_data = np.loadtxt(dataset, dtype = float, delimiter = ",") #dataset read and stored in variable
    
    labels = raw_data[:,-1] #dataset labels furthest right column of dataset specified
    data = raw_data[:,:-1].astype(float) #dataset specified of all columns except the last one 
    

    train_data,test_data,train_labels,test_labels = train_test_split(data, labels, train_size = .75, test_size = .25) #training at 80% and testing at 20% split
    #print("The train Data:", train_data)
    #print("The Labels:", train_labels)



    return train_data,test_data,train_labels,test_labels 

dataset_csv = ["000306853_1.csv", "000306853_2.csv", "000306853_3.csv", "000306853_4.csv"]

analyze_data = load_dataset(dataset_csv[0])  #for dataset 1
trained_data = np.array(analyze_data[0]) 
trained_labels = np.array(analyze_data[2])
test_data = np.array(analyze_data[1])
test_labels = np.array(analyze_data[3])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trained_data,trained_labels)
predictions = clf.predict(test_data)
correct = (predictions == test_labels).sum()
approximate_percentage = correct/len(predictions)*100 

mlp = MLPClassifier(hidden_layer_sizes=[], max_iter=50, tol = .0006, learning_rate_init = 0.05) #no hidden layer sizes put, as dataset 1 appears to be linearly seperable
mlp.fit(trained_data, trained_labels)

predict_mlp = mlp.predict(test_data)
the_accuracy = accuracy_score(test_labels, predict_mlp)

print("File:", dataset_csv[0])
print("============")
print("Decision Tree: %", approximate_percentage, " accuracy")
print("Number of Epochs Used:",mlp.n_iter_)
print("Accuracy: %", the_accuracy * 100)
print(trained_data.shape)



analyze_data = load_dataset(dataset_csv[1])  #for dataset 2
trained_data = np.array(analyze_data[0]) 
trained_labels = np.array(analyze_data[2])
test_data = np.array(analyze_data[1])
test_labels = np.array(analyze_data[3])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trained_data,trained_labels)
predictions = clf.predict(test_data)
correct = (predictions == test_labels).sum()
approximate_percentage = correct/len(predictions)*100

mlp = MLPClassifier(hidden_layer_sizes=[], max_iter=80, tol = .0006, learning_rate_init = 0.05) #no hidden layer sizes put, as dataset 1 appears to be linearly seperable
mlp.fit(trained_data, trained_labels)

predict_mlp = mlp.predict(test_data)
the_accuracy = accuracy_score(test_labels, predict_mlp)


print("File:", dataset_csv[1])
print("============")
print("Decision Tree: %", approximate_percentage, " accuracy")
print("Number of Epochs Used:",mlp.n_iter_)
print("Accuracy: %", the_accuracy * 100)
print(trained_data.shape)



analyze_data = load_dataset(dataset_csv[2])  #for dataset 3
trained_data = np.array(analyze_data[0]) 
trained_labels = np.array(analyze_data[2])
test_data = np.array(analyze_data[1])
test_labels = np.array(analyze_data[3])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trained_data,trained_labels)
predictions = clf.predict(test_data)
correct = (predictions == test_labels).sum()
approximate_percentage = correct/len(predictions)*100

mlp = MLPClassifier(hidden_layer_sizes=[43], max_iter=80, tol = .0006, learning_rate_init = 0.05) #43 neurons in one hidden layer, 80 iterations maximum
mlp.fit(trained_data, trained_labels)

predict_mlp = mlp.predict(test_data)
the_accuracy = accuracy_score(test_labels, predict_mlp)


print("File:", dataset_csv[2])
print("============")
print("Decision Tree: %", approximate_percentage, " accuracy")
print("Number of Epochs Used:",mlp.n_iter_)
print("Accuracy: %", the_accuracy * 100)
print(trained_data.shape)


analyze_data = load_dataset(dataset_csv[3])  #for dataset 4
trained_data = np.array(analyze_data[0]) 
trained_labels = np.array(analyze_data[2])
test_data = np.array(analyze_data[1])
test_labels = np.array(analyze_data[3])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trained_data,trained_labels)
predictions = clf.predict(test_data)
correct = (predictions == test_labels).sum()
approximate_percentage = correct/len(predictions)*100

mlp = MLPClassifier(hidden_layer_sizes=[35], max_iter=80, tol = .0005, learning_rate_init = 0.05) #35 neurons in 1st hidden layer, max iteration set to 80, tol increased by .0001
mlp.fit(trained_data, trained_labels)

predict_mlp = mlp.predict(test_data)
the_accuracy = accuracy_score(test_labels, predict_mlp)


print("File:", dataset_csv[3])
print("============")
print("Decision Tree: %", approximate_percentage, " accuracy")
print("Number of Epochs Used:",mlp.n_iter_)
print("Accuracy: %", the_accuracy * 100)
print(trained_data.shape)




last_dataset = ["Aligned.csv"]
for i in range(1): 
    analyze_data = load_dataset(last_dataset[i])  #UCI dataset
    trained_data = np.array(analyze_data[0]) 
    trained_labels = np.array(analyze_data[2])
    test_data = np.array(analyze_data[1])
    test_labels = np.array(analyze_data[3])

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trained_data,trained_labels)
    predictions = clf.predict(test_data)
    correct = (predictions == test_labels).sum()
    approximate_percentage = correct/len(predictions)*100

    mlp = MLPClassifier(hidden_layer_sizes=[100], max_iter=80, learning_rate = 'adaptive', learning_rate_init = 0.05) #100 neurons in 1st hidden layer, max iteration set to 80, and learning rate set to adaptive
    mlp.fit(trained_data, trained_labels)

    predict_mlp = mlp.predict(test_data)
    the_accuracy = accuracy_score(test_labels, predict_mlp)


    print("File:", last_dataset[i])
    print("============")
    print("Decision Tree: %", approximate_percentage, " accuracy")
    print("Number of Epochs Used:",mlp.n_iter_)
    print("Accuracy: %", the_accuracy * 100)
    print(trained_data.shape)

