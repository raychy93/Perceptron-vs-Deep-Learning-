import numpy as np
import sys
from sklearn.model_selection import train_test_split
#np.set_printoptions(threshold=sys.maxsize)
"""
Name: Rayhan Chowdhury


Based on the results, it is difficult to suggest which dataset is linearly seperable 
as opposed to not linearly seperable. It doesn't take many epochs to get 100% in the training data, and in some cases, the testing data as well.
I set the iterations to just one epoch to assess how each data-set performed, and based on several runs, it seems 000306853_1.csv training data outperforms the other
training data sets slightly in accuracy. Though this might only suggest to me that it converges earlier than the others. Ultimately, based on results, all of the data seems to converge at some point, which suggests to me the results tell me that
most of them are linearly seperable. 

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


def perceptron(trainingSet, target, the_weights, the_threshold, learning_rate):
    """ This method is an implementation of the perceptron learning algorithm """
    pass

    total_accuracy = 0 #placeholder for last recorded accuracy of last executed epoch 

    for x in range(100): #this for loop iterates over an epoch
        
        output = [] #output is stored in an list
        count = 0 #count initialized
        accuracy = 0 #accuracy initialized
        activation = np.sum(the_weights * trainingSet, axis=1) #activation calculated from multiplying the weights and the training set, then summing every row
        
        #print("Epoch#: ", x)
        
        for a in range(len(activation)): #loop where every activation will be compared with threshold
            
            if activation[a]>the_threshold: #if activation in any iteration is greater than threshold, the output list will append 1
                output.append(1)
            elif activation[a]<the_threshold: #if activation in any iteration is less than threshold, the output list will append 0
                output.append(0)

        for b in range(len(output)): 
            if output[b] == target[b]: #if any iteration of outputs is equal to the target label
                count = count + 1      #then increment the count

        accuracy = count/len(output) * 100 #accuracy calculated 
        total_accuracy = accuracy       #total accuracy takes the value of local scope accuracy 
        

        for a in range(len(output)):
            if output[a] < target[a]: #if any iteration of output is lower than target

                the_weights[a] = the_weights[a] + (trainingSet[a] * learning_rate) #single example of weights that are related to that output will be added with the related training set multiplied by learning rate
                the_threshold = the_threshold - learning_rate #threshold goes down by learning rate
 
            elif output[a] > target[a]: #else if any iteration of output is greater than target

                the_weights[a] = the_weights[a] - (trainingSet[a] * learning_rate) #single example of weights that are related to that output will be added with the related training set multiplied by learning rate
                the_threshold = the_threshold + learning_rate #threshold increases by learning rate

                
        
        if(accuracy == 100): #if accuracy reaches 100, break
            break
    
    return total_accuracy, the_weights, the_threshold #return accuracy, weights at final epoch, and last updated threshold
              

for i in range(4): 
    analyze_data = load_dataset(dataset_csv[i])  #loop iterates over 4 datasets and initialized to variable
    trained_data = np.array(analyze_data[0]) 
    trained_labels = np.array(analyze_data[2])
    test_data = np.array(analyze_data[1])
    test_labels = np.array(analyze_data[3])

    the_training_weights = np.random.uniform(-1,1,trained_data.shape) #random weights between -1 and 1 that correspond to same shape as train data
    
    the_testing_weights = np.random.uniform(-1,1,test_data.shape) #random weights between -1 and 1 that correspond to same shape as test data
    the_threshold =  np.random.rand(1) #threshold random number
  
    learning_rate = np.random.uniform(0.01,0.1) #learning rate small random number between .01 and .1
  
    print(" ")
    print(" ")
    perceptron_results = perceptron(trained_data,trained_labels, the_training_weights, the_threshold, learning_rate)
    print("PERCEPTRON TRAINING RESULTS FOR ", dataset_csv[i])
    print("=================================================")
    print((dataset_csv[i]),": ", round(perceptron_results[0], 2), "%, W:", perceptron_results[1], "T:", perceptron_results[2] )
    print(" ")
    print(" ")
    print("PERCEPTRON TESTING RESULTS FOR ", dataset_csv[i])
    print("=================================================")
    perceptron_results = perceptron(test_data,test_labels, the_testing_weights, the_threshold, learning_rate)
    print(dataset_csv[i],": ", round(perceptron_results[0],2), "%, W:", perceptron_results[1], "T:", perceptron_results[2] )
    print(" ")
    print(" ")
