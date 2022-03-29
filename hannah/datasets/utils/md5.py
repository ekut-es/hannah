import hashlib
import pathlib


def generate_file_md5(filename, blocksize=2**10):
    filename = pathlib.Path(filename)

    m = hashlib.md5()
    with filename.open("rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()
