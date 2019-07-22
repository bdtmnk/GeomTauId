from glob import glob
import numpy as np
import ConfigParser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.ini',
                        help="Configuration file")
parser.add_argument('--output', default='validation.txt')
parser.add_argument('--num', default=1)
args = parser.parse_args()
configuration_name = args.config
out_file_name = args.output
num = int(args.num)
config = ConfigParser.RawConfigParser()
config.read(args.config)

TEST_DATA = config.get("data","test")

if __name__ == "__main__":
    file_list = glob(TEST_DATA + "/[DW]*.root")
    np.random.shuffle(file_list)
    if num > -1:
        file_list = file_list[:num]
    out_file = open(out_file_name, 'w')
    for file_name in file_list:
        out_file.write(file_name.split('/')[-1] + '\n')
    out_file.close()
