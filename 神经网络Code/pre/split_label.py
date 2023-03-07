
import os
import random


def check_file_ext(fileList,obj_ext):
    for filename in fileList:
        fn, ext = os.path.splitext(filename)
        if ext != obj_ext:
            return False
    return True


if __name__ == '__main__':

    # 首先把此文件放到三个文件夹的同目录
    # 其次，input1_folder input2_folder label_folder 三个文件夹对应名填入
    # 最后，save_file 为所对应的标签文件

    test_ratio = 0.2
    train_ratio = 1 - test_ratio

    total_file = 'label.txt'

    test_file = 'label_test.txt'
    train_file = 'label_train.txt'

    with open(total_file,'r') as f:
        samplesList = f.readlines()

    random.shuffle(samplesList)

    samplesNum = len(samplesList)

    testSamples = samplesList[:int(samplesNum*test_ratio)]
    trainSamples = samplesList[int(samplesNum*test_ratio):]

    print(f'The number of testSamples is {len(testSamples)}. \n The number of trainSamples is {len(trainSamples)}.')

    with open(test_file,'w') as f:
        f.writelines(testSamples)

    with open(train_file,'w') as f:
        f.writelines(trainSamples)

    print(f'{test_file} and {train_file} have constructed !')