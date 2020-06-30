import os

convert_files = []
for path, subdirs, files in os.walk("./vad_data_balanced/"):
    for name in files:
        if name.endswith("wav") and not name.startswith("."):
            os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 " + os.path.join(path, name))
