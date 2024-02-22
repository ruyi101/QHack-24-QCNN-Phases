# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import numpy as np
from sklearn.model_selection import train_test_split


def data_load_and_process(dataset, binary  = True):
    
    if binary == True:
        if dataset == "Ising":
            dataframe = np.load("Ising_dataset.npz")
            X, y = dataframe['ground_states'], dataframe['phases']
        elif dataset == 'SPT':
            dataframe = np.load("spt_dataset.npz")
            X, y = dataframe['ground_states'], dataframe['phases']
            
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state = 1108)
        
    else:
        if dataset == "Ising":
            dataframe = np.load("Ising_dataset.npz")
            X, y = dataframe['ground_states'], dataframe['magnetization']
        elif dataset == 'SPT':
            dataframe = np.load("spt_dataset.npz")
            X, y = dataframe['ground_states'], dataframe['orderParams']
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state = 1108)
        
        if dataset == 'Ising':
            Y_test = np.where(abs(Y_test)<0.05, 0, 1)
        elif dataset =='SPT':
            Y_test = np.where(abs(Y_test)<0.05, 0, 1)
        
    
    
    return X_train, X_test, Y_train, Y_test

