import os

if __name__ == '__main__':
    file_list = './preprocess/validation_split.txt'
    with open(file_list, 'rb') as fr:
        lines = fr.readlines()
        for line in lines:
            print line.rstrip('\r\n')