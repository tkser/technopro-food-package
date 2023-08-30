import os
import shutil
import zipfile


def exract_zip_file(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)


def remove_macosx_dir(dir_path):
    macosx_dir_path = os.path.join(dir_path, '__MACOSX')
    if os.path.exists(macosx_dir_path):
        shutil.rmtree(macosx_dir_path, ignore_errors=True)


def main():
    input_data_dir = os.path.join(os.path.dirname(__file__), './data/input')

    train_zip_path = os.path.join(input_data_dir, 'train.zip')
    test_zip_path = os.path.join(input_data_dir, 'test.zip')

    if not os.path.exists(train_zip_path) or not os.path.exists(test_zip_path):
        print("Error: train.zip and/or test.zip not found in the specified directory.")
        return

    exract_zip_path = os.path.join(input_data_dir, 'images')
    train_extract_path = os.path.join(exract_zip_path, 'train')
    test_extract_path = os.path.join(exract_zip_path, 'test')

    if not os.path.exists(exract_zip_path):
        os.makedirs(exract_zip_path)

    if not os.path.exists(train_extract_path):
        exract_zip_file(train_zip_path, exract_zip_path)
        print("Extracted train.zip to {}".format(train_extract_path))
    else:
        print("train.zip already extracted to {}".format(train_extract_path))
    
    if not os.path.exists(test_extract_path):
        exract_zip_file(test_zip_path, exract_zip_path)
        print("Extracted test.zip to {}".format(test_extract_path))
    else:
        print("test.zip already extracted to {}".format(test_extract_path))
    
    remove_macosx_dir(exract_zip_path)
    

if __name__ == '__main__':
    main()