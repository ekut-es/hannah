Preprocessing
=============

Dataset
-------

The dataset enhancement is implemented in module speech_recognition.dataset

The dataset classes perform the following operations:

1. Splitting the WAV-files of a dataset in training, development and test_set
2. Reading of WAV-Files and Labels
3. Resampling of WAV-Files to desired frequency
4. If the dataset provides background noise files, background noise is added to the   audios
5. Caching of preprocessed audios
6. Running of the feature extraction
7. Caching of extracted features
8. Provides an interface to the features and associated labels suitable for
   training and evaluation

The dataset preprocessing has the following command line interface:

 +---------------------------------+------------------------------------+
 |  Parameter                      | Default                            | 
 +=================================+====================================+ 
 | --test-pct TEST_PCT             |  10				|
 +---------------------------------+------------------------------------+ 
 | --silence-prob SILENCE_PROB     |  0.1				|
 +---------------------------------+------------------------------------+ 
 | --dev-pct DEV_PCT               |  10				|
 +---------------------------------+------------------------------------+ 
 | --data-folder DATA_FOLDER       |  datasets/speech_commands_v0.02/	|
 +---------------------------------+------------------------------------+ 
 | --noise-prob NOISE_PROB         |  0.8				|
 +---------------------------------+------------------------------------+ 
 | --no-extract-loudest            |  True				|
 +---------------------------------+------------------------------------+ 
 | --timeshift-ms TIMESHIFT_MS     |  100				|
 +---------------------------------+------------------------------------+ 
 | --unknown-prob UNKNOWN_PROB     |  0.1				|
 +---------------------------------+------------------------------------+ 
 | --samplingrate SAMPLINGRATE     |  16000				|
 +---------------------------------+------------------------------------+ 
 | --train-pct TRAIN_PCT           |  80				|
 +---------------------------------+------------------------------------+ 
 | --wanted-words WANTED_WORDS     |  see command line			|
 +---------------------------------+------------------------------------+ 
 | --use-default-split             |  False				|
 +---------------------------------+------------------------------------+ 
 | --input-length INPUT_LENGTH     |  16000				|
 +---------------------------------+------------------------------------+ 
 | --loss {cross_entropy,ctc}      |  cross_entropy			|
 +---------------------------------+------------------------------------+ 
 | --no-group-speakers-by-id       |  True				|
 +---------------------------------+------------------------------------+ 

   
   
.. automodule:: speech_recognition.dataset
   :members:
		
Feature Extraction
------------------

We currently support the following preprocessing directives:

1. Mel Frequency Cepstral Coefficients (MFCC) implemented as implemented in honk (mel)
2. Mel Frequency Cepstral Coefficients (MFCC) as implemented in librosa (mfcc)
3. Mel Frequency Spectrum e.g. MFCC with (melspec)
4. Spectrogram as implemented in librosa (spectrogram)
5. RAW-Audio (raw)

Feature extraction has the following command line interface:

+---------------------------------------------+---------+
|Parameter                                    | Default |
+=============================================+=========+
|--freq-max FREQ_MAX                          | 4000    |
+---------------------------------------------+---------+
|--n-mfcc N_MFCC                              |40       |
+---------------------------------------------+---------+
|--stride-ms STRIDE_MS                        |10       |
+---------------------------------------------+---------+
|--window-ms WINDOW_MS                        | 30      | 
+---------------------------------------------+---------+
|--freq-min FREQ_MIN                          |20       |
+---------------------------------------------+---------+
|--n-mels N_MELS                              |      40 |
+---------------------------------------------+---------+
|--features {mel,mfcc,melspec,spectrogram,raw}| mel     |
+---------------------------------------------+---------+

Feature extreaction is implemented in the module speech_recognition.preprocess_audio.

.. automodule:: speech_recognition.process_audio
   :members:
