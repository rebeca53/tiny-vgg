from pathlib import Path
from json import load
from glob import glob
import shutil, os

source_path = os.path.expanduser('~') + '/Documents/second semester/thesis/EuroSAT_RGB/EuroSAT_RGB/'

val_path_preffix =  Path.cwd() / 'data' / 'class_10_val'
dirNames = ['test_images', 'val_images']

# validation
# TODO: get all folders in the source_path to the folders variable
os.chdir(source_path)
folders = list(filter(os.path.isdir, os.listdir(os.curdir)))
print(folders)
for folder in folders:
    print(folder)

    allimg = os.listdir(folder) # list(Path(folder).glob('*.jpg'))  # Getting all the img files
    last = len(allimg)
    print(last)

    count = 0
    print('move to test_images')
    for i in range(last-24,last+1):
        file_name = './' + folder + '/' + folder + '_' + str(i) + '.jpg'
        print(file_name)
        print(os.path.isfile(file_name))
        if not os.path.isfile(file_name):
            print('BAD FILE')
            exit()

        test_images = val_path_preffix / 'test_images'
        print(test_images)
        if not os.path.isdir(test_images):
            print('BAD DIRECTORY')
            exit()

        shutil.copy2(file_name, test_images)
        count = count + 1
    
    print (count)
    
    print('move to val_images')
    for i in range(last-49, last-24):
        file_name = './' + folder + '/' + folder + '_' + str(i) + '.jpg'
        print(file_name)
        print(os.path.isfile(file_name))
        if not os.path.isfile(file_name):
            print('BAD FILE')
            exit()

        val_images = val_path_preffix / 'val_images'
        print(val_images)
        if not os.path.isdir(val_images):
            print('BAD DIRECTORY')
            exit()

        shutil.copy2(file_name, val_images)
        count = count + 1
    
    print (count)