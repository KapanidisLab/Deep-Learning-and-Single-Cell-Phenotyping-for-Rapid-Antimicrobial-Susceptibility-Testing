import os, re

dirpath = r'C:\Users\zagajewski\Desktop\Data_complete_new_convention\Repeat_4_03_04_21'
file_ext = '.json'

date = 210403
expID = 1

if __name__ == '__main__':

    files_to_process = []
    for root,dir,files in os.walk(dirpath):
        for file in files:
            if file.endswith(file_ext):
                entry = (os.path.join(root,file), file, root)
                files_to_process.append(entry)

    for value in files_to_process:

        (path,file,root) = value

        fname = os.path.splitext(file)[0]

        if len(fname.split('_')) == 11:
            continue

        new_name = str(date) + '_' + str(expID) + '_' + fname + file_ext

        new_file = os.path.join(root,new_name)

        os.rename(path,new_file)