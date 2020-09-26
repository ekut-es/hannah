import VADData
import sys

if __name__ == "__main__":
    print("test")
    print(sys.argv)
    vad = None
    kws = None
    if "--vad_small" in sys.argv:
        print("vad small will be generated")
        if vad is None:
            vad = VADData.VADData()
        # vad.get_small_dataset()
    if "--vad_big" in sys.argv:
        print("vad big will be generated")
        if vad is None:
            vad = VADData.VADData()
        # vad.get_big_dataset()
    if "--vad_all" in sys.argv:
        print("VAD will be generated")
        if vad is None:
            vad = VADData.VADData()
        vad.get_small_dataset()
        vad.get_big_dataset()
    if ("--vad_small" in sys.argv) | ("--vad_big" in sys.argv) | ("--vad_all" in sys.argv):
        vad.split_dataset()
        vad.downsample()
