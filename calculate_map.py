import glob
import os


def calculate_map():
    os.chdir('darknet/')
    weights_files = glob.glob('backup/*000.weights')
    weights_files.sort(key=os.path.getmtime)
    print(weights_files)
    for path in weights_files:
        path_without_extension = path.split('.')[0]
        command = "./darknet detector map signals.data yolov3-tiny-signals.cfg {} > {}.map".format(path, path_without_extension)
        os.system(command)


if __name__ == "__main__":
    calculate_map()