import numpy
import os
import csv

cwd = os.getcwd()

with open(cwd + "/data/pima-indians-diabetes.csv") as f:
    reader = csv.reader(f)
    rows = [row for row in reader]












