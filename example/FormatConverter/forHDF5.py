import numpy as np
from tensorflow.keras.models import load_model
import struct
import sys

model_path = sys.argv[1]


def write_weights(f, weights):
    for v_ in weights:
        for w in v_.tolist():
            binary_value = struct.pack('d', w)
            binary_str = ''.join('{:02x}'.format(x) for x in reversed(binary_value))
            f.write('%s ' % binary_str)
        f.write('\n')
    f.write('\n')


def write_biases(f, biases):
    for b in biases.tolist():
        binary_value = struct.pack('d', b)
        binary_str = ''.join('{:02x}'.format(x) for x in reversed(binary_value))
        f.write('%s ' % binary_str)
    f.write('\n\n')


if __name__ == '__main__':
    model = load_model(model_path)

    f_bin = open("model_hdf5.txt", 'w')
    f_bin.write('%d\n' % (len(model.layers)+1))
    f_bin.write('%s ' % model.layers[0].weights[0].shape[0])
    for layer in model.layers:
        f_bin.write('%d ' % layer.weights[1].shape[0])
    f_bin.write('\n\n')

    for layer in model.layers:
        write_weights(f_bin, np.array(layer.weights[0].value()).transpose())
        write_biases(f_bin, np.array(layer.weights[1].value()))

    f_bin.close()