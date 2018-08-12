"""Functions to generate the features from images."""


import cv2
import numpy as np
from skimage.feature import hog
from print_progress import print_progress


def __get_color_features(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()


def __get_hist_features(image, nbins=32, bins_range=(0, 256)):
    if len(image.shape) < 3:
        return np.histogram(image, bins=nbins, range=bins_range)[0]
    if image.shape[2] == 1:
        return np.histogram(image[:, :, 0], bins=nbins, range=bins_range)[0]
    if image.shape[2] == 3:
        hist1 = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        hist2 = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        hist3 = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((hist1[0], hist2[0], hist3[0]))
        # Return the eature vector
        return hist_features


def __get_hog_features(img, orient, pix_per_cell, cell_per_block,
                       vis=False, feature_vec=True):
    hog_output = hog(img, orientations=orient,
                     pixels_per_cell=(pix_per_cell, pix_per_cell),
                     cells_per_block=(cell_per_block, cell_per_block),
                     block_norm='L2-Hys',
                     transform_sqrt=True,
                     visualize=vis, feature_vector=feature_vec)
    if vis:
        return hog_output[0], hog_output[0]
    else:
        return hog_output


def get_features(file_names, settings):
    """Return the features for all images."""
    features = []
    progress_total_lenght = len(file_names)
    progress_step = max((1, progress_total_lenght//200))
    for idx, file_name in enumerate(file_names):
        image = cv2.imread(file_name)

        color_features = []
        if settings['color']['do']:
            color_cspace = settings['color']['cspace']
            color_size = settings['color']['size']
            color_image = cv2.cvtColor(image, color_cspace)
            color_features = __get_color_features(color_image,
                                                  size=color_size)

        hist_features = []
        if settings['hist']['do']:
            hist_cspace = settings['hist']['cspace']
            hist_nbins = settings['hist']['nbins']
            hist_bins_range = settings['hist']['bins_range']
            hist_image = cv2.cvtColor(image, hist_cspace)
            hist_features = __get_hist_features(hist_image,
                                                nbins=hist_nbins,
                                                bins_range=hist_bins_range)

        hog_features = []
        if settings['hog']['do']:
            for cspace in settings['hog']['cspace']:
                for channel in settings['hog']['channel']:
                    h_image = cv2.cvtColor(image, cspace)
                    h = __get_hog_features(h_image[:, :, channel],
                                           settings['hog']['orient'],
                                           settings['hog']['pix_per_cell'],
                                           settings['hog']['cell_per_block'])
                    hog_features.append(h)

            hog_features = np.ravel(hog_features)

        features.append(np.concatenate((color_features,
                                        hist_features,
                                        hog_features)))

        if idx % progress_step == 0 or idx == progress_total_lenght - 1:
            print_progress(idx+1, progress_total_lenght,
                           prefix='Getting features',
                           suffix='complete')
    return features


def get_features_frame(img, scale, ystart, ystop, settings):
    """Return all features for each subwindow."""
    all_features = []
    pix_per_cell = settings['hog']['pix_per_cell']
    cell_per_block = settings['hog']['cell_per_block']

    image_resized = img[ystart:ystop, :, :]
    if scale != 1:
        imshape = image_resized.shape
        image_resized = cv2.resize(image_resized,
                                   (np.int(imshape[1]/scale),
                                    np.int(imshape[0]/scale)))

    # Define blocks and steps as above
    nxblocks = (image_resized.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (image_resized.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog_images = []
    if settings['hog']['do']:
        for cspace in settings['hog']['cspace']:
            for channel in settings['hog']['channel']:
                h_image = cv2.cvtColor(image_resized, cspace)
                h = __get_hog_features(h_image[:, :, channel],
                                       settings['hog']['orient'],
                                       settings['hog']['pix_per_cell'],
                                       settings['hog']['cell_per_block'],
                                       feature_vec=False)
                hog_images.append(h)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            hog_features = []
            for hog_image in hog_images:
                h = hog_image[ypos:ypos+nblocks_per_window,
                              xpos:xpos+nblocks_per_window].ravel()
                hog_features.append(h)
            hog_features = np.ravel(hog_features)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(image_resized[ytop:ytop+window,
                                              xleft:xleft+window], (64, 64))

            # Get color features
            color_features = []
            if settings['color']['do']:
                color_cspace = settings['color']['cspace']
                color_size = settings['color']['size']
                color_image = cv2.cvtColor(subimg, color_cspace)
                color_features = __get_color_features(color_image,
                                                      size=color_size)

            # Get hist features
            hist_features = []
            if settings['hist']['do']:
                hist_cspace = settings['hist']['cspace']
                hist_nbins = settings['hist']['nbins']
                hist_bins_range = settings['hist']['bins_range']
                hist_image = cv2.cvtColor(subimg, hist_cspace)
                hist_features = __get_hist_features(hist_image,
                                                    nbins=hist_nbins,
                                                    bins_range=hist_bins_range)

            f = np.concatenate((color_features,
                                hist_features,
                                hog_features))

            win_draw = np.int(window*scale)
            xtop_left = np.int(xleft*scale)
            ytop_left = np.int(ytop*scale) + ystart
            xbottom_right = xtop_left + win_draw
            ybottom_right = ytop_left + win_draw
            all_features.append(((xtop_left, ytop_left),
                                 (xbottom_right, ybottom_right),
                                 f))

    return all_features
