import math
import librosa
import numpy as np

def calculate_feature_shape(input_length,
                            features="mel",
                            samplingrate=1600,
                            n_mels=40,
                            n_mfcc=40,
                            stride_ms=10,
                            window_ms=10):
    n_fft = (samplingrate * window_ms) // 1000
    hop_length = (samplingrate * stride_ms)  // 1000
        
    if features == "mel" or features == "mfcc":
        width  = math.floor(input_length / hop_length) + 1
        height = n_mfcc

        return (height, width)

    if features == "melspec":
        width  = math.floor(input_length / hop_length) + 1
        height = n_mels
        return (height, width)

    if features == "spectrogram":
        width  = math.floor(input_length / hop_length) + 1
        height = 1 + n_fft // 2
        return (height, width)
    
    else:
        return (1, input_length)
    

def preprocess_audio(data, features='mel',
                     samplingrate=1600,
                     n_mels=40,
                     n_mfcc=40,
                     dct_filters=None, 
                     freq_min=20,
                     freq_max=4000,
                     window_ms = 40,
                     stride_ms = 10):
    hop_length = (samplingrate * stride_ms)  // 1000
    n_fft = (samplingrate * window_ms) // 1000
    if features == "mel":
        if dct_filters is None:
            dct_filters =  librosa.filters.dct(n_mfcc, n_mels)
        data = librosa.feature.melspectrogram(data, sr=samplingrate,
                                              n_mels=n_mels, hop_length=hop_length,
                                              n_fft=n_fft, fmin=freq_min, fmax=freq_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        data = data.transpose()

        
    elif features == "mfcc":
        data = librosa.feature.mfcc(data,
                                    sr=samplingrate,
                                    n_mels=n_mels,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length,
                                    n_fft=n_fft,
                                    fmin=freq_min,
                                    fmax=freq_max)
        data = data.astype(np.float32)

    elif features == "melspec":
        data = librosa.feature.melspectrogram(data, sr=samplingrate,
                                              n_mels=n_mels, hop_length=hop_length,
                                              n_fft=n_fft, fmin=freq_min, fmax=freq_max)
        data = data.astype(np.float32)
    elif features == "spectrogram":
        data = librosa.core.stft(data, hop_length=hop_length,
                                 n_fft=n_fft)
        data = np.abs(data)
        data = data.astype(np.float32)
        
    elif features == "raw":
        data = np.array(data)
        data = np.expand_dims(data, axis=0)
        data = data.astype(np.float32)
    else:
        raise Exception("Unknown feature extractor: {}".format(features))
    
    return data



def main():
    import sys
    from matplotlib import pyplot as plt
    import librosa.display
    
    audio_file  = sys.argv[1]
    sampling_rate = 16000
    audio_data = librosa.core.load(audio_file, sr=sampling_rate)[0]

    feature_set = ["raw", "spectrogram", "melspec", "mfcc", "mel"]
    features = {}
    plt.figure()
    
    for num, feature in enumerate(feature_set):
        data = preprocess_audio(audio_data, features=feature, samplingrate=sampling_rate)

        features[feature] = data
        
        plt.subplot(len(feature_set), 1, num+1)
        if feature == "raw":
            librosa.display.waveplot(data[0], sr=sampling_rate)
        else:
            librosa.display.specshow(data,
                                     y_axis='log',
                                     x_axis='time',
                                     sr=sampling_rate)
           

        plt.title(feature)

    print(features["mel"] - features["mfcc"])
        
    plt.show()
        


if __name__ == "__main__":
    main()
