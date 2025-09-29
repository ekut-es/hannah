# Post-Processing with Viterbi Decoding

This implementation for the Post-Processing of CNN evaluations with a Hidden Markov Model (HMM) and Viterbi decoding in HANNAH is based on <cite>**Precise Localization within the GI Tract** [1]</cite> and <cite>**Energy-efficient Seizure Detection** [2]</cite>. The datasets used in those two publications, Video Capsule Endoscopy (VCE) datasets (Rhode Island and Galar) and EEG datasets (CHB-MIT), are supported. However, in general, the main criteria is that the test set is ordered in time and thus, this can be applied to many other applications and datasets as well. For the Rhode Island, the Galar and the CHB-MIT dataset, the ordering is ensured within the dataset files in HANNAH. Additionally, we combined the usage of neural networks with a HMM. After a network is trained on the given datasets, the evaluations of the CNN can be used as a direct input as observations for a HMM and subsequent Viterbi Decoding.

Next, the Viterbi Decoding computes the most likely sequence of hidden states (or classes in this case) based on the provided observations given by the neural network. For applications such as the VCE, a HMM in combination with Viterbi Decoding is a very reasonable choice as there is some preknowledge about this application. For example, we can leverage the knowledge in which order the capsule traverses the gastrointestinal tract. Furthermore, we know that the capsule cannot move back to a preceding organ after it has entered the next one. This can be encoded in the transition probabilities of the HMM.

As a first step, one needs to invoke a training with HANNAH for one of the supported datasets. This generates two output files named as ``{model_name}_cnn_train_output`` and ``{model_name}_cnn_train_output``. Both are CSV Files, containing for each input one row, where the columns indicate the study id, the CNN evaluation and the true label (e.g. `100,0,1` for study ID=100, prediction of CNN = 0, true label = 1). Subsequently, they can be read into ``hmm_window.py`` (which allows to compute the Viterbi algorithm with a predefined window size).


Thus, the Viterbi Decoding can be invoked by:

	python hannah/sequential_analysis/hmm/hmm_window.py --cnn_output_dir '/PATH/TO/TRAINED_MODELS/DIR/' --class_of_interest 2 --quantization False

Parameters that can be specified are:

`cnn_output_dir`
: type = str, Path to the output of the trained CNN.

`model_name`
: default = "mobilenetv3_small_075", Name of the CNN used for training.

`window_size`
: default = 300, Window size used during Viterbi Decoding.

`quantization`
: default = False, If desired to validate on Hardware, the matrices should be quantized.

`quant_type`
: default = rounding, only needs to be specified if quantization is True. Besides ``rounding``, ``linear_scaling`` is an option.

`window_size_sweep`
: default = False, type=bool, Whether to perform a window size sweep first.

`class_of_interest`
: type = int, must be provided - the class number for which the delay should be tracked, e.g. 1 for seizure detection, 2 for small intestine in RI dataset, 3 for small intestine in Galar dataset.

This reads the CSV files into Pandas Dataframes with the following columns ``["id", "preds", "labels"]`` for both the train and the test set.
The CNN output evaluated on the train set is used to compute the confusion matrix for the train set. This naturally encodes the emission probabilities. Thus, this is used to generate the emission matrix for the HMM. 

## Window size
The precise value of this parameter is not too crucial for the discussed settings. However, if one considers the final hardware architecture, this should be kept in mind. With the given files, a window size search can be performed, if the window_size_sweep parameter is set to True. It performs a simple grid search by running the Viterbi decoding with the same samples for different window sizes. The window sizes are currently encoded in ``window_size_sweep.py`` and can be adjusted. The file outputs two additional plots ``boxplot_window_sweep.pdf`` and ``window_size_sweep.pdf``. 


## Visualization 
For all test studies, one final confusion matrix is generated as well as a plot with the total error percentages. Furthermore, for a single study (default is the first one of the test set), the predictions for the CNN only, the true labels and the Viterbi predictions are plotted. This serves as a comparison to visualize the flaws and strengths of both methods (or their combination).
All figures are saved to ``hannah/hannah/sequential_analysis/hmm/figures_hmm``.


## Grid Search
A simple grid search can be performed to search for the best transition probabilities (the emission probabalities are directly encoded in the confusion matrix of the CNN evaluated on the train set) by running:

	python hannah/sequential_analysis/hmm/grid_search_trans.py --cnn_output_dir '/PATH/TO/TRAINED_MODELS/DIR/'

Parameters that can be specified are:

`cnn_output_dir`
: type = str, Path to the output of the trained CNN.

`model_name`
: default = "mobilenetv3_small_075", type=str, Name of the CNN used for training.

`values`
: default = [0.9, 0.95, 0.99, 0.999], type=list of floats, A list of possible non-logarithmic values to choose from during the grid search. A Permutation of all combinations is performed

For each combination of transition probabilities, the Viterbi Decoding is performed and the resulting accuracies plotted (``acc_grid_search.pdf``).

**Note:** A setting such as given in VCE studies is assumed here (from one state only transitioning to the succeeding state is possible **and** once the last state is reached, there is no possibility to transfer to another state.)


# Examples
Below, for three patients exemplarily (one of each of the integrated datasets) the predictions of the CNN only vs the combinatorial approach of CNN and HMM/Viterbi decoding are visualized.

### Galar and Rhode Island dataset (VCE - images as input)
<img src="/../assets/VCE_single_patient_Gal.png" alt="Single patient of Galar dataset" width="500"/>   	<img src="/../assets/VCE_single_patient_RI.png" alt="Single patient of Rhode Island dataset" width="500"/>

### CHBMIT dataset (EEG data as input)
<img src="/../assets/CHBMIT_single_patient_HMM.png" alt="Single patient of EEG dataset" width="500"/>



# References 

[1] Werner, J., Gerum, C., Reiber, M., Nick, J., & Bringmann, O. (2023, October). Precise localization within the GI tract by combining classification of CNNs and time-series analysis of HMMs. In International Workshop on Machine Learning in Medical Imaging (pp. 174-183). Cham: Springer Nature Switzerland.

[2] Werner, J., Kohli, B., Bernardo, P. P., Gerum, C., & Bringmann, O. (2024, June). Energy-Efficient Seizure Detection Suitable for Low-Power Applications. In 2024 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.