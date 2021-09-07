from pandas import read_csv
from pandas import concat
import pandas as pd
import os

from pandas.core.frame import DataFrame

# get all filename
test_filename, real_filename = [], []
months = [1, 7, 11]
days = [1, 7, 14, 21]
for i in months:
    for j in days:
        test_filename.append('test-max-%d-%d.csv' % (i, j))
        real_filename.append('actual-max-%d-%d.csv' % (i, j))

# combine filename data


def concat_data(in_name, out_name):
    print(in_name)
    li = []
    index = 0
    for i in in_name:
        if index == 0:
            df = read_csv('dataset/%s' % i)
        else:
            df = read_csv('dataset/%s' % i, header=0)
        li.append(df)
    index += 1
    total = concat(li, axis=0, ignore_index=True)
    print(total)
    df_fn = DataFrame(total)
    df_fn.to_csv('dataset/%s.csv'%out_name, index=0)

# main
concat_data(test_filename,'testdata')
concat_data(real_filename,'realdata')
