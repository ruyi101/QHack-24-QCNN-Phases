import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding


def data_embedding(X, embedding_type='Statevector'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Statevector':
        qml.QubitStateVector(X, wires = range(8))