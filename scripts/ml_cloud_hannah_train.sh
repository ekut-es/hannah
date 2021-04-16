source /opt/poetry_env/speech-recognition-*/bin/activate 
python3 -m speech_recognition.train trainer.max_epochs=2 module.num_workers=12
