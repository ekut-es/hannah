""" This module implements feature extraction from raw audio data """

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
    """Calculates the shape of the given features
    
    Parameters
    ----------
      input_length : int
        lenght of the input in samples
      features : str
        The selected feature to extract. Currently one of: 
        - raw: for RAW single channel audiodata
        - mel: Mel Frequency Cepstral Coefficients
        - mfcc: Alternative implementation of Mel Frequency Cepstral Coefficients
        - melspec: Mel-scaled spectrogram
        - spectrogram: Spectrogram
      n_mels: int
        Number of frequency bands in melspectrogram
      n_mfcc: int
        Number of mfcc/mel features after dct
      stride_ms: int
        Stride of the feature extractor in ms
      window_ms: int
        Size of the feature window in ms

    Returns: (int, int)
       Shape of the features (N,T). T is temporal dimension, N is number of channels.
"""
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
    """Calculates the features for a given audio

     Args:
      input_length : int
        lenght of the input in samples
      features : str
        The selected feature to extract. Currently one of: 
        - raw: for RAW single channel audiodata
        - mel: Mel Frequency Cepstral Coefficients
        - mfcc: Alternative implementation of Mel Frequency Cepstral Coefficients
        - melspec: Mel-scaled spectrogram
        - spectrogram: Spectrogram
      dct_filters: np.ndarray 
        Dct filter coefficients to transform mel scaled spectrograms into 
        MFCC features. 
      n_mels: int
        Number of frequency bands in melspectrogram
      n_mfcc: int
        Number of mfcc/mel features after dct
      stride_ms: int
        Stride of the feature extractor in ms
      window_ms: int
        Size of the feature window in ms

    Returns: np.ndarray (NxT)
      Extracted features N is channel dimenstion, T is Temporal dimension
"""
    
    hop_length = (samplingrate * stride_ms)  // 1000
    n_fft = (samplingrate * window_ms) // 1000
    if features == "mel":
        if dct_filters is None:
            dct_filters =  librosa.filters.dct(n_mfcc, n_mels)
        data = librosa.feature.melspectrogram(data, sr=samplingrate,
                                              n_mels=n_mels, hop_length=hop_length,
                                              n_fft=n_fft, fmin=freq_min, fmax=freq_max, center=False)
        data[data > 0] = np.log(data[data > 0])

        data = np.matmul(dct_filters, data)         
        data = data.astype(np.float32)
        

        
    elif features == "mfcc":
        data = librosa.feature.mfcc(data,
                                    sr=samplingrate,
                                    n_mels=n_mels,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length,
                                    n_fft=n_fft,
                                    fmin=freq_min,
                                    fmax=freq_max,
                                    center=False)
        data = data.astype(np.float32)

    elif features == "melspec":
        data = librosa.feature.melspectrogram(data, sr=samplingrate,
                                              n_mels=n_mels, hop_length=hop_length,
                                              n_fft=n_fft, fmin=freq_min, fmax=freq_max, center=False)
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
