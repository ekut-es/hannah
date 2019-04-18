import math
import librosa
import numpy as np

def calculate_feature_shape(input_length,
                            features="mel",
                            samplingrate=1600,
                            n_dct_filters=40,
                            stride_ms=10):
    if features == "mel":
        hop_length = (samplingrate * stride_ms)  // 1000
        height = math.floor(input_length / hop_length) + 1
        width = n_dct_filters

        return (height, width)

    else:
        return (1, input_length)
    

def preprocess_audio(data, features='mel', samplingrate=1600, n_mels=40, dct_filters=None, 
                     freq_min=20, freq_max=4000, window_ms = 10, stride_ms = 10):
    hop_length = (samplingrate * stride_ms)  // 1000
    n_fft = (samplingrate * window_ms) // 1000
    if features == "mel":
        data = librosa.feature.melspectrogram(data, sr=samplingrate,
                                              n_mels=n_mels, hop_length=hop_length,
                                              n_fft=n_fft, fmin=freq_min, fmax=freq_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        
    elif features == "raw":
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32)
    else:
        raise Exception("Unknown feature extractor: {}".format(features))
    
    return data

