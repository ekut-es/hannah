def next_power_of2(x):
    return 1 << (x - 1).bit_length()
