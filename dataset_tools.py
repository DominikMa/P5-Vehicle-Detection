"""Functions to handle the dataset."""
from os import path, makedirs
import csv
import cv2
import numpy as np
from glob import glob
from sklearn.utils import shuffle
from print_progress import print_progress

car_folder = 'vehicles'
non_car_folder = 'non-vehicles'
label_files = [('data/object-detection-crowdai/labels_crowdai.csv', ','),
               ('data/object-dataset/labels.csv', ' ')]


def prepare_dataset():
    """Generate the 64x64 images of cars and non-cars out of the datasets."""
    for label_file in label_files:
        label_file_path = label_file[0]
        label_file_delimiter = label_file[1]

        # Create folders for cars and non-cars
        dirname = path.dirname(label_file_path)
        car_path = path.join(dirname, car_folder)
        non_car_path = path.join(dirname, non_car_folder)
        if not path.exists(car_path):
            makedirs(car_path)
        if not path.exists(non_car_path):
            makedirs(non_car_path)

        print("Preparing dataset " + dirname)
        frames = {}
        # Load csv and iterate through each line
        with open(label_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile,
                                delimiter=label_file_delimiter,
                                quotechar='"')
            # Get sorting of headers
            headers = reader.__next__()
            idx_frame = headers.index('frame')
            idx_xmin = headers.index('xmin')
            idx_ymin = headers.index('ymin')
            idx_xmax = headers.index('xmax')
            idx_ymax = headers.index('ymax')
            idx_label = headers.index('label')

            # Get all boxes in a dict for each frame
            for row in reader:
                frame = row[idx_frame]
                xmin, ymin = row[idx_xmin], row[idx_ymin]
                xmax, ymax = row[idx_xmax], row[idx_ymax]
                label = row[idx_label]
                if int(xmax)-int(xmin) > 1900:
                    print("Skipped a box")
                    continue

                if frame not in frames:
                    frames[frame] = []

                frames[frame].append({'xmin': int(xmin),
                                      'ymin': int(ymin),
                                      'xmax': int(xmax),
                                      'ymax': int(ymax),
                                      'label': label.lower()})

        print("Found " + str(len(frames)) + " frames")
        progress_total_lenght = len(frames)
        progress_step = max((1, progress_total_lenght//500))
        car_images_counter = 0

        for idx, (frame, boxes) in enumerate(frames.items()):
            frame_name, frame_ext = path.splitext(frame)

            car_boxes = list(filter(lambda box: box['label'] == 'car',
                                    boxes))

            image = cv2.imread(path.join(dirname, frame))
            car_images = __get_car_images(image, car_boxes)
            for idx_car, car_image in enumerate(car_images):
                cv2.imwrite(path.join(car_path,
                                      frame_name+"_"+str(idx_car)+frame_ext),
                            car_image)
            car_images_counter += len(car_images)

            non_car_images = __get_non_car_images(image, car_boxes)
            for idx_car, non_car_image in enumerate(non_car_images):
                cv2.imwrite(path.join(non_car_path,
                                      frame_name+"_"+str(idx_car)+frame_ext),
                            non_car_image)
            if idx % progress_step == 0 or idx == progress_total_lenght - 1:
                print_progress(idx+1, progress_total_lenght,
                               prefix='Preparing dataset',
                               suffix='complete')
        print('Generated {:d} car and non-car images'
              .format(car_images_counter))


def __get_car_images(image, car_boxes):
    car_images = []
    for car_box in car_boxes:
        car_image = image[car_box['ymin']:car_box['ymax']+1,
                          car_box['xmin']:car_box['xmax']+1]
        car_images.append(cv2.resize(car_image, (64, 64)))
    return car_images


def __get_non_car_images(image, car_boxes):
    non_car_images = []
    image_h, image_w = image.shape[:2]

    for car_box in car_boxes:
        box_h = car_box['ymax']-car_box['ymin']
        box_w = car_box['xmax']-car_box['xmin']
        new_box = __get_nonoverlapping_box(image_h, image_w,
                                           box_h, box_w,
                                           car_boxes)
        if new_box is None:
            continue
        non_car_image = image[new_box['ymin']:new_box['ymax']+1,
                              new_box['xmin']:new_box['xmax']+1]
        non_car_images.append(cv2.resize(non_car_image, (64, 64)))
    return non_car_images


def __get_nonoverlapping_box(image_h, image_w,
                             box_h, box_w,
                             car_boxes):

    resize_counter = 0
    counter = 0
    while True:
        if resize_counter > 50:
            print('Failed to find a nonoverlapping box')
            return None
        if counter > 50:
            counter = 0
            box_h = int(box_h*0.80)
            box_w = int(box_w*0.80)
            resize_counter += 1
        try:
            new_x_min = np.random.randint(0, image_w-box_w)
        except ValueError:
            print('Error while searching for nonoverlapping box')
            print('image_w:', image_w)
            print('box_w:', box_w)
            print()
            continue

        try:
            new_y_min = np.random.randint(image_h//4,
                                          image_h-box_h)
        except ValueError:
            new_y_min = np.random.randint(0, image_h-box_h)
        new_box = {'xmin': new_x_min,
                   'ymin': new_y_min,
                   'xmax': new_x_min+box_w,
                   'ymax': new_y_min+box_h}

        overlapping = False
        for car_box in car_boxes:
            if __boxes_overlap(car_box, new_box, image_h, image_w):
                overlapping = True
                break
        if not overlapping:
            return new_box
        counter += 1


def __boxes_overlap(first_box, second_box, image_h, image_w):
    image_zero = np.zeros((image_h, image_w))
    image_zero[first_box['ymin']:first_box['ymax']+1,
               first_box['xmin']:first_box['xmax']+1] += 1
    image_zero[second_box['ymin']:second_box['ymax']+1,
               second_box['xmin']:second_box['xmax']+1] += 1

    if np.max(image_zero) > 1:
        return True
    else:
        return False


def get_data(folder_names, train_test_split=0.4):
    """Return the filenames of all car and non car images."""
    x_train, x_test = [], []
    y_train, y_test = (), ()
    for folder_name in folder_names:
        for sub_folder in glob(path.join(folder_name,
                                         car_folder,
                                         '**/'),
                               recursive=True):
            car_files = []
            car_files.extend(glob(path.join(sub_folder,
                                            '*.png')))
            car_files.extend(glob(path.join(sub_folder,
                                            '*.jpg')))
            car_files.sort()
            first_split = int((train_test_split/2.0)*len(car_files))
            second_split = int((1-(train_test_split/2.0))*len(car_files))
            train_car_files = car_files[first_split:second_split]
            test_car_files = car_files[:first_split]
            test_car_files.extend(car_files[second_split:])
            x_train.extend(train_car_files)
            y_train += (np.ones(len(train_car_files)),)
            x_test.extend(test_car_files)
            y_test += (np.ones(len(test_car_files)),)

        for sub_folder in glob(path.join(folder_name,
                                         non_car_folder,
                                         '**/'),
                               recursive=True):
            non_car_files = []
            non_car_files.extend(glob(path.join(sub_folder,
                                                '*.png')))
            non_car_files.extend(glob(path.join(sub_folder,
                                                '*.jpg')))
            non_car_files.sort()

            first_split = int((train_test_split/2.0)*len(non_car_files))
            second_split = int((1-train_test_split/2.0)
                               * len(non_car_files))
            train_non_car_files = non_car_files[first_split:second_split]
            test_non_car_files = non_car_files[:first_split]
            test_non_car_files.extend(non_car_files[second_split:])
            x_train.extend(train_non_car_files)
            y_train += (np.zeros(len(train_non_car_files)),)
            x_test.extend(test_non_car_files)
            y_test += (np.zeros(len(test_non_car_files)),)

    y_train = np.concatenate(y_train)
    y_test = np.concatenate(y_test)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test
