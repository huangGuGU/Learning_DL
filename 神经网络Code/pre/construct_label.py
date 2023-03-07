import sys
import os



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

    input1_folder = '1' #

    label_folder = '2'

    save_file = 'label.txt'

    input1_imgList = os.listdir(input1_folder)

    label_imgList = os.listdir(label_folder)

    print(f'The number of input1 is: {len(input1_imgList)}, '

          f'the number of labels is: {len(label_imgList)}.')

    samplesList = []
    for input1_imgIns in input1_imgList:
            if input1_imgIns in label_imgList:
                samplesList.append('pre/{} pre/{}\n'.format(os.path.join(input1_folder, input1_imgIns),
                                                       os.path.join(label_folder, input1_imgIns)))
            else:
                sys.exit()


    with open(save_file,'w') as f:
        f.writelines(samplesList)

    print(f'{save_file} has constructed !')