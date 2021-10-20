import os
import numpy as np

file_b = open("base.txt", "r")
file_fs_v = open("fs_val.txt", "r")
file_fs_t = open("fs_test.txt", "r")

lines_b = file_fs_t.read().rstrip("\n").split(',')

print(lines_b)