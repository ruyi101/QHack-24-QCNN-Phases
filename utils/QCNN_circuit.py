import pennylane as qml
from utils import unitary
from utils import embedding

# Quantum Circuits for Convolutional layers
def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
def conv_layer2(U, params):
    U(params, wires=[0, 6])
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])
def conv_layer3(U, params):
    U(params, wires=[0,4])

# Quantum Circuits for Pooling layers
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires=[i + 1, i])
def pooling_layer2(V, params):
    V(params, wires=[2,0])
    V(params, wires=[6,4])
def pooling_layer3(V, params):
    V(params, wires=[0,4])



def QCNN_structure(U, params, U_params):
    
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]
    param4 = params[3 * U_params: 3 * U_params + 2]
    param5 = params[3 * U_params + 2: 3 * U_params + 4]
    param6 = params[3 * U_params + 4: 3 * U_params + 6]

    # Pooling Ansatz1 is used by default
    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param4)
    conv_layer2(U, param2)
    pooling_layer2(unitary.Pooling_ansatz1, param5)
    conv_layer3(U, param3)
    pooling_layer3(unitary.Pooling_ansatz1, param6)
    
    
    
    
def double_QCNN_structure(U, params, U_params):
    num_params_layer = 3 * U_params + 6
    for i in range(2):
        params_temp = params[i * num_params_layer: (i+1) * num_params_layer]
        
    
        param1 = params[0:U_params]
        param2 = params[U_params: 2 * U_params]
        param3 = params[2 * U_params: 3 * U_params]
        param4 = params[3 * U_params: 3 * U_params + 2]
        param5 = params[3 * U_params + 2: 3 * U_params + 4]
        param6 = params[3 * U_params + 4: 3 * U_params + 6]

        # Pooling Ansatz1 is used by default
        conv_layer1(U, param1)
        pooling_layer1(unitary.Pooling_ansatz1, param4)
        conv_layer2(U, param2)
        pooling_layer2(unitary.Pooling_ansatz1, param5)
        conv_layer3(U, param3)
        pooling_layer3(unitary.Pooling_ansatz1, param6)
        
def quadraple_QCNN_structure(U, params, U_params):
    num_params_layer = 3 * U_params + 6
    for i in range(4):
        params_temp = params[i * num_params_layer: (i+1) * num_params_layer]
        
    
        param1 = params[0:U_params]
        param2 = params[U_params: 2 * U_params]
        param3 = params[2 * U_params: 3 * U_params]
        param4 = params[3 * U_params: 3 * U_params + 2]
        param5 = params[3 * U_params + 2: 3 * U_params + 4]
        param6 = params[3 * U_params + 4: 3 * U_params + 6]

        # Pooling Ansatz1 is used by default
        conv_layer1(U, param1)
        pooling_layer1(unitary.Pooling_ansatz1, param4)
        conv_layer2(U, param2)
        pooling_layer2(unitary.Pooling_ansatz1, param5)
        conv_layer3(U, param3)
        pooling_layer3(unitary.Pooling_ansatz1, param6)
    
    


def QCNN_structure_without_pooling(U, params, U_params):
    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params]

    conv_layer1(U, param1)
    conv_layer2(U, param2)
    conv_layer3(U, param3)

def QCNN_1D_circuit(U, params, U_params):
    param1 = params[0: U_params]
    param2 = params[U_params: 2*U_params]
    param3 = params[2*U_params: 3*U_params]

    for i in range(0, 8, 2):
        U(param1, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(param1, wires=[i, i + 1])

    U(param2, wires=[2,3])
    U(param2, wires=[4,5])
    U(param3, wires=[3,4])



dev = qml.device('default.qubit', wires = 8)
# dev = qml.device('lightning.gpu', wires = 8)
@qml.qnode(dev)
def QCNN(X, params, U, U_params, embedding_type='Statevector', cost_fn='cross_entropy', nums_layer = 1):


    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)
    for layer in range(nums_layer):
        # Quantum Convolutional Neural Network
        if U == 'U_TTN':
            QCNN_structure(unitary.U_TTN, params, U_params)
        elif U == 'U_5':
            QCNN_structure(unitary.U_5, params, U_params)
        elif U == 'U_6':
            QCNN_structure(unitary.U_6, params, U_params)
        elif U == 'U_9':
            QCNN_structure(unitary.U_9, params, U_params)
        elif U == 'U_13':
            QCNN_structure(unitary.U_13, params, U_params)
        elif U == 'U_14':
            QCNN_structure(unitary.U_14, params, U_params)
        elif U == 'U_15':
            QCNN_structure(unitary.U_15, params, U_params)
        elif U == 'U_SO4':
            QCNN_structure(unitary.U_SO4, params, U_params)
        elif U == 'U_SU4':
            QCNN_structure(unitary.U_SU4, params, U_params)
        elif U == 'double_SU4':
            double_QCNN_structure(unitary.U_SU4, params, U_params)
        elif U == 'q_SU4':
            quadraple_QCNN_structure(unitary.U_SU4, params, U_params)
        elif U == 'U_SU4_no_pooling':
            QCNN_structure_without_pooling(unitary.U_SU4, params, U_params)
        elif U == 'U_SU4_1D':
            QCNN_1D_circuit(unitary.U_SU4, params, U_params)
        elif U == 'U_9_1D':
            QCNN_1D_circuit(unitary.U_9, params, U_params)
        else:
            print("Invalid Unitary Ansatze")
            return False

    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(4))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=4)
    return result
