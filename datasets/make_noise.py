
import librosa
import os
from shutil import copyfile, move

def split_noise_files(noise_dir):
    new_noise_dir = noise_dir + "chunks_"
    os.mkdir(new_noise_dir)
    chunk_list = []
    c = 0
    for f in os.listdir(noise_dir):
        if f.endswith("wav"):
           for i in range(60):
               audio, sr = librosa.core.load(os.path.join(noise_dir, f), offset=i, duration=1.0)
               output_path = os.path.join(new_noise_dir, "chunk{0}.wav".format(c))  
               librosa.output.write_wav(output_path, audio, sr)
               c = c+1

def add_noise_to_testing_list():
    noise_files = [item for item in os.listdir("speech_commands_v0.02/_background_noise_chunks_")]
    testing_list = open("speech_commands_v0.02/testing_list.txt", "a")
    for noise_file in noise_files:
        testing_list.write(os.path.join("_background_noise_chunks_", noise_file) + os.linesep)
    testing_list.close()

def merge_classes():
    all_classes_dir ="./speech_commands_v0.02/all"
   # os.mkdir(all_classes_dir)
    subdirs = [item for item in os.listdir("speech_commands_v0.02") if os.path.isdir(os.path.join("speech_commands_v0.02", item))]
    for subdir in subdirs:
       if (not "noise" in subdir) and (not "all" in subdir):
          originaldir = os.path.join("speech_commands_v0.02", subdir)
          list_with_files = [f for f in os.listdir(originaldir)]
          files_to_copy = list_with_files[:200]
          for file in files_to_copy:
              copyfile(os.path.join(originaldir, file), os.path.join(all_classes_dir, file))

def get_test_files():
   # testing_list = open("speech_commands_v0.02/testing_list.txt", "r#    os.mkdir("test_files")
      testing_list = [line.rstrip('\n') for line in open("speech_commands_v0.02/testing_list.txt")]  
      for test_file in testing_list:
          print(os.path.basename(test_file))
          if not "chunk" in test_file:
             copyfile(os.path.join("speech_commands_v0.02", test_file), os.path.join("test_files", os.path.basename(test_file)))
    


def move_dirs_for_eval():
   new_dir = "dataset_eval"
   move("./speech_commands_v0.02/_background_noise_chunks_", os.path.join(new_dir, "test/noise"))
   move("./speech_commands_v0.02/all",  os.path.join(new_dir, "test/speech"))
   
def main():
   dir = "./speech_commands_v0.02"
   noise_dir = os.path.join(dir, "_background_noise_")
#   split_noise_files(noise_dir)
#   add_noise_to_testing_list()
   get_test_files()
#   merge_classes()
#   move_dirs_for_eval()


if __name__ == "__main__":
   main()
