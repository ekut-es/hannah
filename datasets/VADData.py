import os


class VADData():

    big_noise = [
      "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip",
      "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip",
      "https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip",
      "https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_test.zip",
      "https://zenodo.org/record/2529934/files/FSDnoisy18k.audio_train.zip",
      "https://zenodo.org/record/2529934/files/FSDnoisy18k.meta.zip"]
    big_noise_folder = ["./noise_files/FSDKaggle/", "./noise_files/FSDKaggle/",
                        "./noise_files/FSDKaggle/", "./noise_files/FSDnoisy/",
                        "./noise_files/FSDnoisy/", "./noise_files/FSDnoisy/"]

    #big_speech = ["https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5.1-2020-06-22/en.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/de.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/fr.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/es.tar.gz",
    #              "https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/it.tar.gz"]
    big_speech = ["https://cdn.commonvoice.mozilla.org/cv-corpus-5.1-2020-06-22/it.tar.gz"]

    def __init__(self):
        print("works")
        self.make_structure()

    def downsample(self):
        print("Start with downsample of VAD")
        for path, subdirs, files in os.walk("./vad_data_balanced/"):
            for name in files:
                if name.endswith("wav") and not name.startswith("."):
                    os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 -loglevel quiet " + os.path.join(path, "new" + name))
                    os.system("rm " + os.path.join(path, name))
                    os.system("mv " + os.path.join(path, "new" + name) + " " + os.path.join(path, name))
                elif name.endswith("mp3") and not name.startswith("."):
                    os.system("ffmpeg -y -i " + os.path.join(path, name) + " -ar 16000 -ac 1 -loglevel quiet " + os.path.join(path, name.replace(".mp3", ".wav")))
                    os.system("rm " + os.path.join(path, name))
        print("Finished Downsample")

    def make_structure(self):
        os.system("mkdir -p ./noise_files")
        os.system("mkdir -p ./speech_files")
        os.system("mkdir -p ./vad_data_balanced/")

    def download(self, link, certificate=""):
        os.system("wget " + link + " " + certificate)

    def unzip(self, file, destination):
        os.system("unzip -P pass " + file + " -d " + destination)

    def untar(self, file, destination):
        os.system("tar -xvzf " + file + " -C " + destination)

    def get_small_dataset(self):
        for i in range(1, 10):
            self.download("https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio." + str(i) +".zip")
            pass
        self.unzip("'*.zip'", "./noise_files/")
        os.system("rm *.zip")

        self.download("https://zeos.ling.washington.edu/corpora/UWNU/uwnu-v2.tar.g", "--no-check-certificate")
        self.untar("uwnu-v2.tar.gz", "./speech_files/")

    def get_big_dataset(self):
        for link, folder  in zip(self.big_noise, self.big_noise_folder):
            self.download(link)
            self.unzip("*.zip", folder)
            os.system("rm *.zip")

        for link in self.big_speech:
            self.download(link)
            self.untar("*.tar.gz", "./speech_files/")
            os.system("rm *.tar.gz")
            os.system("mv ./speech_files/cv-corpus-5.1-2020-06-22/ ./speech_files/mozilla/")
