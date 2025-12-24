from scipy.signal import convolve2d
from lib.API import tensor, optimized
from PIL import Image
import numpy as np
import pickle
import os

_tensor = tensor()
_optimized = optimized()

filters = [
    [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    [[1, 1, 1], [1, -8, 1], [1, 1, 1]],
    [[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]],
    [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]],
    [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]],
    [[2, -1, 0], [-1, 2, -1], [0, -1, 2]],
    [[0, -1, 2], [-1, 2, -1], [2, -1, 0]],
    [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
    [[1, -2, 1], [-2, 5, -2], [1, -2, 1]],
    [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
    [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]],
    [[1, 1, 0], [1, -4, 0], [0, 0, 0]],
    [[0, -1, -2], [1, 1, -1], [2, 1, 0]],
]

def process_image(input_image):
    if isinstance(input_image, Image.Image):
        img = input_image.resize((10, 10)).convert("RGB")
    else:
        img = Image.open(input_image).resize((10, 10)).convert("RGB")
    arr = np.array(img) / 255.0
    channels = (arr[:, :, 0], arr[:, :, 1], arr[:, :, 2])
    feature_maps = []
    for kernel in filters:
        feature_map = []
        for channel in channels:
            out = convolve2d(channel, kernel, mode='same', boundary='fill', fillvalue=0)
            feature_map.append(out.copy())
        feature_maps.append(np.sum(np.stack(feature_map, axis=0), axis=0))
    feature_maps = np.stack(feature_maps, axis=0)
    feature_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1]//2, 2, feature_maps.shape[2]//2, 2).max(axis=(2, 4))
    flat_fmaps = feature_maps.ravel()
    return flat_fmaps

def load_dataset(dataset_path, mpc):
    input_data = []
    input_labels = []
    classes = os.listdir(dataset_path)
    for label, cls in enumerate(classes):
        k = 0
        for image in os.listdir(f"{dataset_path}/{cls}"):
            if k == mpc:
                break
            k += 1
            flat_fmaps = process_image(f"{dataset_path}/{cls}/{image}")
            input_data.append(flat_fmaps)
            input_labels.append(label)
    classes_num = len(classes)
    return input_data, input_labels, classes, classes_num, np.eye(classes_num)[input_labels]

def forward(inp, w, b, layers_num):
    inp = (inp - inp.mean()) / inp.std()
    a = np.array([None] * (layers_num+1))
    z = np.array([None] * layers_num)
    i = 0
    a[0] = inp
    while (i < layers_num):
        z_i = _tensor.sum(_tensor.dot(a[i], w[i]), b[i])
        z[i] = z_i
        if i == layers_num-1:
            a[i+1] = _optimized.softmax(z_i)
        else:
            a[i+1] = _optimized.ReLU(z_i)
        i += 1
    return a[layers_num], a, z

def backward(a, z, ytrue, yhat, w, layers_num):
    dw = []
    db = []
    dz = [yhat - ytrue]
    i = layers_num-1
    while (i > -1):
        dw_i = np.outer(a[i], dz[0])
        dw.insert(0, dw_i)
        db.insert(0, dz[0])
        if i > 0:
            da_prev = _tensor.dot(dz[0], w[i].T)
            dz_prev = da_prev * (z[i-1] > 0)
            dz[0] = dz_prev
        i-= 1
    return dw, db

def predict(image, w, b, layers_num, classes):
    inp = process_image(image)
    yhat, a, z = forward(inp, w, b, layers_num)
    if -np.sum(yhat * np.log(yhat + 1e-12)) > 1.5:
        return None
    return classes[np.argmax(yhat)]

def save(path, w, b, classes, input_labels, layers):
    model = [
        w,
        b,
        classes,
        input_labels,
        layers
    ]
    pickle.dump(model, open(path, "wb"))

def load(path):
    model = pickle.load(open(path, "rb"))
    return model[0], model[1], model[2], model[3], model[4]

def evaluate(dataset_path, w, b, layers_num, classes):
    total = 0
    for cls in os.listdir(dataset_path):
        i = 0
        x = 0
        s = 0
        for image in os.listdir(f"{dataset_path}/{cls}"):
            if x == 1000:
                break
            if predict(f"{dataset_path}/{cls}/{image}", w, b, layers_num, classes) == cls:
                s += 1
            x += 1
        i += 1
        print(f"Number {cls}: {s/1000}")
        total += s/1000

    print("Total:", total/10)