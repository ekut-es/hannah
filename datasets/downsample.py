import os

for path, subdirs, files in os.walk("./vad_data_balanced/"):
    for name in files:
        if name.endswith("wav") and not name.startswith("."):
            os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 " + os.path.join(path, "new" + name))
            os.system("rm " + os.path.join(path, name))
            os.system("mv " + os.path.join(path, "new" + name) + " " + os.path.join(path, name))
