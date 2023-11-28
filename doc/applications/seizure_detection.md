# Dataset creation

The current used dataset was generated with the 16 channels with the highest variance within the ictal data from the <cite>**CHB-MIT Scalp EEG Database** [1]</cite>. If a new preprocessed dataset should be created, the ``scripts/eeg/eeg_dataset_creator.py`` file can be used. This takes the edf files from the CHB-MIT dataset as an input and performs a basic preprocessing including a band-pass filtering (0.1 Hz and 50  Hz) to remove DC components as well as the noisy component from the EEG measurement device. It finally creates binary labels for each data fragment.

Parameters that can be specified are amongst others:

class_ratio
: default = 5, Ratio of zeros:ones in the final `dev` and `retrain` datasets to account for the scarcity of the ictal data

data_length
: default = 1, Number of seconds per data point. For a sample rate of 256, the final data will be of the shape (D,C,256xdata_length)

samp_rate
: default = 256, Sampling rate for the dataset. The data is already sampled at 256 Hz, use this argument only when the rate needed differs from this

The dataset creation can be invoked by:

    python eeg_dataset_creator.py --output_dir "./data/" --class_ratio 4 --data_length 0.5

Currently, the `16c_retrain_id` preprocessed dataset is configured as the default dataset in HANNAH if someone works with the CHB-MIT dataset. This was generated with a `class_ratio` of 4, a `data_length` of 0.5 and a `sampling_rate` of 256.

[1] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.


# Training a base model with the CHB-MIT dataset

The training with the preprocessed CHB-MIT dataset can be invoked with:
	
    hannah-train dataset=chbmit features=identity ~normalizer

The data is normalized directly when loaded and does not need additional normalization during training. `~normalizer` is used to turn additional normalization off. Adding `+dataset.weighted_loss=True` improves the results notably (same applies for the retraining). The trained model can function as a base model which can be fine-tuned for each patient by subsequent retraining.


# Retraining

The main idea of retraining is to account for individual differences within the data. To perform subsequent retraining, for each patient a new model needs to be trained based on the checkpoint of the best model trained on all patients. The prior trained base model is loaded and retrained on patient-specific data. To invoke this training the dataset `chbmitrt` needs to be used as this specifically loads patient-specific data only. For a single patient, the retraining can be invoked by:

	hannah-train dataset=chbmitrt trainer.max_epochs=10 model=tc-res8 module.batch_size=8 input_file=/PATH/TO/BASE_MODEL/best.ckpt ~normalizer

Alternatively, if the retraining should be performed for all patients, ``scripts/eeg/run_patient_retraining_complete.py`` can be used. To execute this script, one can use

    python run_patient_retraining_complete.py 'model-name' 'dataset_name' 

A model which has been successfully used for this application is for example the TC-ResNet8 `tc-res8` and it is recommended to use `16c_retrain_id` as a dataset name, which names the preprocessd CHB-MIT dataset with balanced class samples. During the retraining, for each patient a results folder is generated.