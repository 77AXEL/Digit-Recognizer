#include <math.h>

extern "C" __declspec(dllexport)
void ReLU(float* inp, int length) {
    for (int i = 0; i < length; i++) {
        inp[i] = fmaxf(0.0f, inp[i]);
    }
}

extern "C" __declspec(dllexport)
void softmax(float* inp, float* exp_shifted, int length) {
    float max = inp[0];
    for (int i = 1; i < length; i++) {
        if (inp[i] > max) max = inp[i];
    }
    float sum = 0.0f;
    const float CLIP = 88.0f;
    for (int i = 0; i < length; i++) {
        float shifted = inp[i] - max;
        if (shifted > CLIP) shifted = CLIP;
        if (shifted < -CLIP) shifted = -CLIP;
        exp_shifted[i] = expf(shifted);
        sum += exp_shifted[i];
    }
    for (int i = 0; i < length; i++) {
        exp_shifted[i] /= sum;
    }
}

extern "C" __declspec(dllexport)
float loss(float* yhat, float* ytrue, int length) {
    float sum = 0.0f;
    const float EPS = 1e-7f;
    for (int i = 0; i < length; i++) {
        float p = yhat[i];
        if (p < EPS) p = EPS;
        if (p > 1.0f) p = 1.0f;
        sum += -ytrue[i] * logf(p);
    }
    return sum / length;
}

extern "C" __declspec(dllexport)
double regularize(double w_mean, double ld_val) {
    if (w_mean < 0.01) {
        ld_val *= 1.5;
    }
    else if (w_mean > 0.05) {
        ld_val *= 0.5;
    }
    ld_val = fmin(fmax(ld_val, 1e-6), 0.01);
    return ld_val;
}

extern "C" __declspec(dllexport)
double update(
    double** w, double** dw,
    double** b, double** db,
    int* layers,
    int nl,
    double lr,
    double ld
) {
    double w_mean = 0.0;
    long total_weights = 0;
    for (int L = 0; L < nl; L++) {
        int rows = layers[L];
        int cols = layers[L + 1];
        long size = (long)rows * (long)cols;
        for (long i = 0; i < size; i++) {
            double reg_term = ld * w[L][i];
            w[L][i] -= lr * (dw[L][i] + reg_term);
            w_mean += fabs(w[L][i]);
            total_weights++;
        }
        for (int i = 0; i < cols; i++) {
            b[L][i] -= lr * db[L][i];
        }
    }
    if (total_weights > 0)
        w_mean /= total_weights;
    return w_mean;
}