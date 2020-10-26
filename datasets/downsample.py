import os

print("Start working")
for path, subdirs, files in os.walk("./vad_data_balanced/"):
    for name in files:
        if name.endswith("wav") and not name.startswith("."):
            os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 -loglevel quiet " + os.path.join(path, "new" + name))
            os.system("rm " + os.path.join(path, name))
            os.system("mv " + os.path.join(path, "new" + name) + " " + os.path.join(path, name))
        elif name.endswith("mp3") and not name.startswith("."):
            os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 -ac 1 -loglevel quiet " + os.path.join(path, name.replace(".mp3", ".wav")))
            os.system("rm " + os.path.join(path, name))
print("Finished")

