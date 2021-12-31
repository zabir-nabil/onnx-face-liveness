import numpy as np
import tf2onnx
import onnxruntime as rt
import cv2

def predict(img, model_path = "face_liveness.onnx"):
    if img.shape != (112, 112, 3):
        return -1

    dummy_face = np.expand_dims(np.array(img, dtype=np.float32), axis = 0) / 255.

    providers = ['CPUExecutionProvider']
    m = rt.InferenceSession(model_path, providers=providers)
    onnx_pred = m.run(['activation_5'], {"input": dummy_face})
    print(onnx_pred)
    liveness_score = list(onnx_pred[0][0])[1]

    return liveness_score

fake_face_1 = cv2.resize(cv2.imread('ronaldo.png'), (112, 112)) 
fake_face_2 = cv2.resize(cv2.imread('Print_1.png'), (112, 112)) 

live_face_1 = cv2.resize(cv2.imread('facecam.jpg'), (112, 112)) 
live_face_2 = cv2.resize(cv2.imread('blur.png'), (112, 112))

ff1s = predict(fake_face_1)
ff2s = predict(fake_face_2)

lf1s = predict(live_face_1)
lf2s = predict(live_face_2)

print("fake scores:")
print(ff1s)
print(ff2s)

print("--------------------")

print("live scores:")
print(lf1s)
print(lf2s)



