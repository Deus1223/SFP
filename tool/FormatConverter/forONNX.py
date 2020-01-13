import onnx
from onnx import numpy_helper
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
    onnx_model = onnx.load(model_path)
    params = onnx_model.graph.initializer

    weights = []
    biases = []
    cnt = 0
    for p in params:
        if (cnt%2) == 0:
            weights.insert(0, numpy_helper.to_array(p).transpose())
        else:
            biases.insert(0, numpy_helper.to_array(p).transpose())
        cnt+=1

    f_bin = open("model_onnx.txt", 'w')
    f_bin.write('%d\n' % (int(cnt/2)+1))
    f_bin.write('%s ' % weights[0].shape[1])
    for idx in range(int(cnt/2)):
        f_bin.write('%d ' % biases[idx].shape[0])
    f_bin.write('\n\n')

    for idx in range(int(cnt/2)):
        write_weights(f_bin, weights[idx])
        write_biases(f_bin, biases[idx])

    f_bin.close()