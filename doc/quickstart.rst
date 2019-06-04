Quickstart
==========


Training
--------
Training on CPU can be invoked by:

.. code-block:: bash
		
    python3.6 -m speech_recognition.train  --no_cuda  --model ekut-raw-cnn3-relu

Training on 1st GPU can be invoked by:

.. Code-block:: bash
		
    python3.6 -m speech_recognition.train  --gpu_no 0  --model ekut-raw-cnn3-relu

Trained models are saved under trained_models/model_name .

Evaluation
----------

To run only the evaluation of a model use:

.. code-block:: bash

    python3.6 -m speech_recognition.train --no_cuda --model ekut-raw-cnn3-relu --batch_size 256 --input_file trained_models/ekut-raw-cnn3-relu/model.pt --type eval

