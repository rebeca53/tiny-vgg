from pathlib import Path
import os
import random
import json
import shutil

# n01882714, n02165456, n02509815, n03662601, n04146614, n04285008, n07720875, n07747607, n07873807, n07920052
# path = Path.cwd() / 'data' / 'class_10_train' / 'n07920052' / 'images'  # The path to use
# to_usenum = 0  # Start num
# allimg = list(path.glob('*.jpg'))  # Getting all the img files
# chdir(path)  # Changing the cwd to the path

# for i, imgfile in enumerate(allimg):
#     to_usename = f"n07920052_{to_usenum+i}.JPEG"  # The name to use
#     imgfile.rename(to_usename) 

def renameAllTrain():
    classCodes = ['n01882714', 'n02165456', 'n02509815', 'n03662601', 'n04146614', 'n04285008', 'n07720875', 'n07747607', 'n07873807', 'n07920052'];
    for classCode in enumerate(classCodes):
        renameTrainImages(classCode[1])

def renameTrainImages(className):
    prev = Path.cwd()
    path = Path.cwd() / 'data' / 'class_10_train' / className / 'images'  # The path to use
    to_usenum = 0  # Start num
    allimg = list(path.glob('*.jpg'))  # Getting all the img files
    os.chdir(path)  # Changing the cwd to the path

    for i, imgfile in enumerate(allimg):
        to_usename = f"{className}_{to_usenum+i}.JPEG"  # The name to use
        imgfile.rename(to_usename) 

    os.chdir(prev)

# renameAllTrain()


# # Preparing validation and test images
# def copy_files():
#     # euroSat paths
#     origin_path = Path.home() / 'Documents' / 'second semester' / 'thesis' / 'EuroSAT_RGB' / 'EuroSAT_RGB'
    
#     # Iterate over files in directory
#     # https://www.geeksforgeeks.org/python-loop-through-folders-and-files-in-directory/
#     for path, folders, files in os.walk(origin_path):    
#         # List contain of folder
#         for folder_name in folders:
#             print(f"Content of '{folder_name}'")
#             # List content from folder
#             print(os.listdir(f"{path}/{folder_name}"))
#             copy_files
#             print()
    
#         break

def renameValImages(dirName, startNumber, val_dict):
    prev_cwd = Path.cwd()
    path = Path.cwd() / 'data' / 'class_10_val' / dirName  # The path to use
    to_usenum = startNumber  # Start num
    # allimg = list(path.glob('*.JPEG'))  # Getting all the img files
    allimg = list(path.glob('*.jpg'))  # Getting all the img files

    os.chdir(path)  # Changing the cwd to the path
    random.shuffle(allimg) # shuffle the list of images so they will not be sorted by class

    for i, imgfile in enumerate(allimg):
        filename = os.path.basename(imgfile)
        # print(filename)
        to_usename = f"val_{to_usenum}.JPEG"  # The name to use
        update_val_dict(val_dict, filename, to_usename)
        # print(to_usename)
        imgfile.rename(to_usename)
        to_usenum += 1

    os.chdir(prev_cwd)

def createDictEntry(fileName):
    preffix = str(fileName).split("_")[0] # returns ('AnnualCrop', '_1.jpg')
    # print(preffix)
    if preffix == 'AnnualCrop':
        return  {
            "class": "n01882714",
            "description": "annual crop",
            "index": 5
        }
    elif preffix == 'Forest':
        return {
            "class": "n02165456",
            "description": "forest",
            "index": 1
        }
    elif preffix == 'HerbaceousVegetation':
        return {
            "class": "n02509815",
            "description": "herbaceous vegetation",
            "index": 7
        }
    elif preffix == 'Highway':
        return {
            "class": "n03662601",
            "description": "highway",
            "index": 0
        }
    elif preffix == 'Industrial':
        return {
            "class": "n04146614",
            "description": "industrial",
            "index": 4
        }
    elif preffix == 'Pasture':
        return {
            "class": "n04285008",
            "description": "pasture",
            "index": 9
        }
    elif preffix == 'PermanentCrop':
        return {
            "class": "n07720875",
            "description": "permanent crop",
            "index": 3
        }
    elif preffix == 'Residential':
        return {
            "class": "n07747607",
            "description": "residential",
            "index": 8
        }
    elif preffix == 'River':
        return {
            "class": "n07873807",
            "description": "river",
            "index": 2
        }
    elif preffix == 'SeaLake':
        return {
            "class": "n07920052",
            "description": "sea, lake, sea or lake",
            "index": 6
        }
    else:
        print("Failed to create dict entry. Invalid preffix in the file name.")

    # match preffix:
    #     case 'AnnualCrop':
    #         return  {
    #             "class": "n01882714",
    #             "description": "annual crop",
    #             "index": 5
    #         }
    #     case 'Forest':
    #         return {
    #             "class": "n02165456",
    #             "description": "forest",
    #             "index": 1
    #         }
    #     case 'HerbaceousVegetation':
    #         return {
    #             "class": "n02509815",
    #             "description": "herbaceous vegetation",
    #             "index": 7
    #         }
    #     case 'Highway':
    #         return {
    #             "class": "n03662601",
    #             "description": "highway",
    #             "index": 0
    #         }
    #     case 'Industrial':
    #         return {
    #             "class": "n04146614",
    #             "description": "industrial",
    #             "index": 4
    #         }
    #     case 'Pasture':
    #         return {
    #             "class": "n04285008",
    #             "description": "pasture",
    #             "index": 9
    #         }
    #     case 'PermanentCrop':
    #         return {
    #             "class": "n07720875",
    #             "description": "permanent crop",
    #             "index": 3
    #         }
    #     case 'Residential':
    #         return {
    #             "class": "n07747607",
    #             "description": "residential",
    #             "index": 8
    #         }
    #     case 'River':
    #         return {
    #             "class": "n07873807",
    #             "description": "river",
    #             "index": 2
    #         }
    #     case 'SeaLake':
    #         return {
    #             "class": "n07920052",
    #             "description": "sea, lake, sea or lake",
    #             "index": 6
    #         }
    #     case _:
    #         print("Failed to create dict entry. Invalid preffix in the file name.")

def update_val_dict(dict, fileName, newname):
    # print(createDictEntry(fileName))
    dict[newname] = createDictEntry(fileName)
    print(dict[newname])

def write_val_dict(val_dict):
    sorted_dict = dict(sorted(val_dict.items()))
    with open('result.json', 'w') as fp:
        json.dump(sorted_dict, fp)

def renameAllVal():
   val_dict = {}
   renameValImages('test_images', 0, val_dict)
   renameValImages('val_images', 250, val_dict)
   write_val_dict(val_dict)

# renameAllVal()

def renameMapImages():
    map_dict = {}
    dirName = 'map_images'
    # renameValImages('map_images', 500, map_dict)

    prev_cwd = Path.cwd()
    path = Path.cwd() / 'data' / 'class_10_val' / dirName  # The path to use

    # allimg = list(path.glob('*.JPEG'))  # Getting all the img files
    allimg = list(path.glob('*.jpg'))  # Getting all the img files

    os.chdir(path)  # Changing the cwd to the path
    random.shuffle(allimg) # shuffle the list of images so they will not be sorted by class

    for i, imgfile in enumerate(allimg):
        filename = os.path.basename(imgfile)
        # print(filename)
        # Image2 = filename.replace("-1", "-1_s", 1)
        to_usename = filename.replace('.jpg','.JPEG', 1)  # The name to use
        update_val_dict(map_dict, filename, to_usename)
        # print(to_usename)
        imgfile.rename(to_usename)

    os.chdir(prev_cwd)

    write_val_dict(map_dict)

renameMapImages()