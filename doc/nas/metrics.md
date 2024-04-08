# Performance Metrics in Neural Architecture Search

Currently there are two main sources of Performance Metrics used in the hannah's NAS subystem. 



1. The main training loop generates performance metrics using the training loops, these metrics are logged using the lightning logging system during training and are then extracted using the 'HydraOptCallback', and are only available for optimization purposes after a training has been run. These kinds of metrics are also generated for normal training runs. 
2. Estimators can provide metrics before the neural networks have been trained. Predictors are used in presampling phases of the neural architecture search. Predictors are not and will not be used outside of neural architecture search. 

There are 2 subclasses of predictors. 
   - Machine Learning based predictors: These predictors provide an interface based on: `predict`, `update`, `load`, `train`
   - Analytical predictors, the interface of these methods only contains the: `predict`

The current implementation has a few problems:

- not using a unified interface for both predictors induces breakage at a lot of places in the nas flow
- it is currently not possible to configure more than one predictor at the same time, which has led to things like hardcoding additional predictors in the NAS loops: https://es-git.cs.uni-tuebingen.de/es/ai/hannah/hannah/-/blob/main/hannah/nas/search/search.py?ref_type=heads#L136
- Currently it is not immediately clear how device metrics should be generated, and especially how device metrics obtained from device execution should be generated. 

There have been a few approaches to this. 
1. The BackendPredictor, it instantiates a backend and then calls the predict method on the untrained model https://es-git.cs.uni-tuebingen.de/es/ai/hannah/hannah/-/blob/main/hannah/nas/performance_prediction/simple.py?ref_type=heads#L42
2. Target Specific estimators, like this mlonmcu predictor: https://es-git.cs.uni-tuebingen.de/es/ai/hannah/hannah/-/merge_requests/378/diffs#a45007495fb172b95977d0692f84cff89bf6d692
