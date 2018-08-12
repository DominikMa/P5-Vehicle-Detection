"""Pipeline for project 5."""

from os import path, listdir
import time
import glob
import pickle
from moviepy.editor import VideoFileClip

import cv2

from dataset_tools import prepare_dataset, get_data
from features import get_features
from classifier import create_classifier
from classification import FrameClassificator

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from P4 import Lanefinder

DEBUG_VARS = {'trim_data': 1}

PREPARE_DATASET = False
PROCESS_IMAGES = True
PROCESS_IMAGES_COMBI = True
PROCESS_MOVIES = True
PROCESS_MOVIES_COMBI = True

DATA_DIRS = [  # 'data/object-dataset/',
             # 'data/object-detection-crowdai',
             'data/vehicle-detection-dataset']

SETTINGS = {
            'feature': {'color': {'do': True,
                                  'cspace': cv2.COLOR_BGR2YCR_CB,
                                  'size': (32, 32)},
                        'hist': {'do': True,
                                 'cspace': cv2.COLOR_BGR2YCR_CB,
                                 'nbins': 16,
                                 'bins_range': (0, 256)},
                        'hog': {'do': True,
                                'cspace': [cv2.COLOR_BGR2YCR_CB],
                                'channel': [0, 1, 2],
                                'orient': 9,
                                'pix_per_cell': 8,
                                'cell_per_block': 2},
                        'save': False,
                        'load': True,
                        'path': 'saved_data/features/',
                        'name': 'features.p'
                        },
            'classifier': {'find_svm': True,
                           'pca_size': 0,
                           'parameter_svm': [{'dual': [True],
                                              'tol': [1e-4],
                                              'max_iter': [1000]
                                              }],
                           'find_dt': True,
                           'parameter_dt': [{   # 'min_samples_split': [100],
                                             'max_depth': [10]
                                             }],
                           'save': False,
                           'load': True,
                           'path': 'saved_data/classifier/',
                           'name': 'classifier.p'},
            'train_test_split': 0.05,
            'verbose': 1
            }


def __load_features():
    features = None
    with open(path.join(SETTINGS['feature']['path'],
                        SETTINGS['feature']['name']), 'rb') as f:

        print("Loading features.")
        features = pickle.load(f)
    return (features['x_train'], features['y_train'],
            features['x_test'], features['y_test'])


def __gen_features():
    # Get train and test data
    print("Loading train and test data")
    x_train, y_train, x_test, y_test = get_data(DATA_DIRS,
                                                SETTINGS['train_test_split'])

    if DEBUG_VARS['trim_data'] is not None:
        x_train = x_train[:len(x_train)//DEBUG_VARS['trim_data']]
        y_train = y_train[:len(y_train)//DEBUG_VARS['trim_data']]
        x_test = x_test[:len(x_test)//DEBUG_VARS['trim_data']]
        y_test = y_test[:len(y_test)//DEBUG_VARS['trim_data']]

    # Get fetures
    t1 = time.time()
    print('Extracting features from the train data')
    x_train = get_features(x_train, SETTINGS['feature'])
    print('Extracting features from the test data')
    x_test = get_features(x_test, SETTINGS['feature'])
    t2 = time.time()
    print('Feature extraction took',
          round(t2-t1, 4),
          'Seconds')

    if SETTINGS['feature']['save']:
        with open(path.join(SETTINGS['feature']['path'],
                            SETTINGS['feature']['name']), 'wb') as f:
            print("Saving features.")
            features = {'x_train': x_train,
                        'y_train': y_train,
                        'x_test': x_test,
                        'y_test': y_test}
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        with open(path.join(SETTINGS['feature']['path'],
                            'settings.p'), 'wb') as f:
            pickle.dump(SETTINGS['feature'], f, pickle.HIGHEST_PROTOCOL)
    return x_train, y_train, x_test, y_test


def __load_classifiers():
    with open(path.join(SETTINGS['classifier']['path'],
                        SETTINGS['classifier']['name']), 'rb') as f:

        print("Loading classifier.")
        classifier = pickle.load(f)
        return (classifier['svm'], classifier['dt'],
                classifier['x_scaler'], classifier['x_pca'])


def __gen_classifiers():
    x_scaler = StandardScaler().fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)

    if SETTINGS['classifier']['pca_size'] > 0:
        x_pca = PCA(n_components=SETTINGS['classifier']['pca_size'])
        x_pca.fit(x_train_scaled)
        x_train_pca = x_pca.transform(x_train_scaled)
    else:
        x_pca = None
        x_train_pca = x_train_scaled

    svm, dt = create_classifier(x_train_pca, x_train_scaled, y_train,
                                SETTINGS['classifier'])
    if SETTINGS['classifier']['save']:
        with open(path.join(SETTINGS['classifier']['path'],
                            SETTINGS['classifier']['name']), 'wb') as f:
            print("Saving classifier.")
            features = {'svm': svm,
                        'dt': dt,
                        'x_scaler': x_scaler,
                        'x_pca': x_pca
                        }
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        with open(path.join(SETTINGS['classifier']['path'],
                            'settings.p'), 'wb') as f:
            pickle.dump(SETTINGS['classifier'], f, pickle.HIGHEST_PROTOCOL)
    return svm, dt, x_scaler, x_pca


# Prepare datasets
if PREPARE_DATASET:
    prepare_dataset()
    print("All datasets prepared")
    print()

# Get train and test features and labels
if SETTINGS['feature']['load']:
    try:
        x_train, y_train, x_test, y_test = __load_features()
    except (KeyError, FileNotFoundError):
        print("Could not load features.")
        x_train, y_train, x_test, y_test = __gen_features()
else:
    x_train, y_train, x_test, y_test = __gen_features()

print()
print('Got ' + str(len(x_train)) + ' train samples.')
print('Got ' + str(len(x_test)) + ' test samples')
print('With ' + str(len(x_train[0])) + ' features.')
print()


# Train classifier
if SETTINGS['classifier']['load']:
    try:
        svm, dt, x_scaler, x_pca = __load_classifiers()
    except (KeyError, FileNotFoundError):
        print("Could not load classifiers.")
        svm, dt, x_scaler, x_pca = __gen_classifiers()
else:
    svm, dt, x_scaler, x_pca = __gen_classifiers()
print()
x_test_scaled = x_scaler.transform(x_test)
if SETTINGS['classifier']['pca_size'] > 0:
    x_test_pca = x_pca.transform(x_test_scaled)
else:
    x_test_pca = x_test_scaled


# Print some stats
if SETTINGS['verbose'] > 0:
    # Print some stats
    if svm is not None:
        t1 = time.time()
        y_svm = svm.predict(x_test_pca)
        t2 = time.time()
        print('Results for SVM (', round((t2-t1)/(len(x_test_pca)/1000), 4),
              'Seconds per 1000 samples):')
        print()
        print(classification_report(y_test, y_svm))
        print()
    if dt is not None:
        t1 = time.time()
        y_dt = dt.predict(x_test_scaled)
        t2 = time.time()
        print('Results for DT (', round((t2-t1)/(len(x_test_scaled)/1000), 4),
              'Seconds per 1000 samples):')
        print(classification_report(y_test, y_dt))
        print()
    print()


scales = [1, 1.8]
y_start = [400, 400]
y_stop = [480, 640]


# Process image and videos
if PROCESS_IMAGES:
    for test_image_name in listdir('test_images/'):
        image = cv2.imread(path.join('test_images/', test_image_name))
        frame_classificator = FrameClassificator(image.shape[: 2],
                                                 SETTINGS['feature'],
                                                 svm, dt, x_scaler, x_pca,
                                                 scales=scales,
                                                 ystart=y_start,
                                                 ystop=y_stop,
                                                 draw_debug=True)
        processed_image = frame_classificator.classify_frame(image)
        cv2.imwrite(path.join('./output_images',
                              test_image_name),
                    processed_image)

if PROCESS_IMAGES_COMBI:
    for test_image_name in listdir('test_images/'):
        image = cv2.imread(path.join('test_images/', test_image_name))
        frame_classificator = FrameClassificator(image.shape[: 2],
                                                 SETTINGS['feature'],
                                                 svm, dt, x_scaler, x_pca,
                                                 scales=scales,
                                                 ystart=y_start,
                                                 ystop=y_stop)
        lanefinder = Lanefinder()
        processed_image = lanefinder.distortion_correction(image)
        processed_image = frame_classificator.classify_frame(processed_image)

        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = lanefinder.process_image(processed_image)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path.join('./output_images',
                              'combi_'+test_image_name),
                    processed_image)

if PROCESS_MOVIES:
    for test_movie_name in glob.glob('*.mp4'):
        clip = VideoFileClip(test_movie_name)
        frame_classificator = FrameClassificator((clip.h, clip.w),
                                                 SETTINGS['feature'],
                                                 svm, dt, x_scaler, x_pca,
                                                 scales=scales,
                                                 ystart=y_start,
                                                 ystop=y_stop,
                                                 convert_color=True,
                                                 draw_debug=True)
        new_clip = clip.fl_image(frame_classificator.classify_frame)
        new_clip.write_videofile(path.join('./output_videos',
                                           test_movie_name),
                                 audio=False)

if PROCESS_MOVIES_COMBI:
    for test_movie_name in glob.glob('*.mp4'):
        clip = VideoFileClip(test_movie_name)
        frame_classificator = FrameClassificator((clip.h, clip.w),
                                                 SETTINGS['feature'],
                                                 svm, dt, x_scaler, x_pca,
                                                 scales=scales,
                                                 ystart=y_start,
                                                 ystop=y_stop,
                                                 convert_color=True)
        lanefinder = Lanefinder()
        new_clip = clip.fl_image(lanefinder.distortion_correction)
        new_clip = new_clip.fl_image(frame_classificator.classify_frame)
        new_clip = new_clip.fl_image(lanefinder.process_image)
        new_clip.write_videofile(path.join('./output_videos',
                                           'combi_'+test_movie_name),
                                 audio=False)
