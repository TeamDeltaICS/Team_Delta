import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("gru_paper.onnx", providers=["CPUExecutionProvider"])

def infer(history):
    x = np.expand_dims(history.astype(np.float32), axis=0) # (1,N,6)
    inp = { sess.get_inputs()[0].name: x }
    y = sess.run(None, inp)[0]
    return y[0].tolist()
