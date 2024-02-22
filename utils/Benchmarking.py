from utils import data
from utils import Training
from utils import QCNN_circuit
from utils import Hierarchical_circuit
import numpy as np

def accuracy_test(predictions, labels, cost_fn, binary = True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)
        else:
            acc = 0
            for l, p in zip(labels, predictions):
                pred = int(abs(p) >= 0.05)
                acc += int(pred == l)
            return acc/len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l,p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)




def Benchmarking(dataset, Unitaries, U_num_params, Embeddings, circuit, cost_fn, nums_layer = 1, binary = True):
    I = len(Unitaries)
    J = len(Embeddings)

    for i in range(I):
        for j in range(J):
            
            U = Unitaries[i]
            U_params = U_num_params[i]
            Embedding = Embeddings[0]
            print('Embedding: ' + Embedding)

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " with " + cost_fn + ", " + str(nums_layer) + " repeated layers")
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit, cost_fn, nums_layer)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn, nums_layer) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
            print("Accuracy for " + U + " :" + str(accuracy))
            f = open('Result/result.txt', 'a')
            f.write("Loss History for " + circuit + " circuits, " + U + " " + " with " + cost_fn + ", " + str(nums_layer) + " repeated layers")
            f.write("\n")
            f.write('Model: ' + dataset)
            f.write("\n")
            f.write('Binary: ' + str(binary))
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " :" + str(accuracy))
            f.write("\n")
            f.write("Parameters:")
            
            f.write(str(trained_params))
            f.write("\n")
            f.write("\n")
            f.write(str(trained_params.tolist()))
            f.write("\n")
            f.write("\n")
            f.close()

