import argparse
import sys
from CoreImageProcessing import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import random
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class VehicleClassifier:
    """ Classifies vehicles and provides methods to train and measure the performance of the classifier."""

    cip = None                             # Core image processing instance
    train_vehicles_path = []               # Training dataset with vehicle samples
    train_non_vehicles_path = []           # Training dataset with non-vehicle samples
    classifier = None                      # Classifier instance (e.g. LinearSVC)
    scaler = None                          # StandardScaler() from sklearn

    # -----------------------------------------------------------------------
    # Feature extraction parameters

    color_space = 'YCrCb'                  # Used color space for feature extraction (RGB, LUV, HLS, YUV and YCrCb)

    # spatial binning
    spatial_binning_enabled = True         # If true spatial binning will be applied during feature extraction
    spatial_size = (32, 32)                # size of spatially binned image

    # channel histogram
    hist_channels_enabled = True           # If true channel histograms will be applied during feature extraction
    hist_nb_bins = 64                      # number if histogram bins

    # HOG configuration
    hog_enabled = True                     # If true HOG features will be applied during feature extraction
    hog_orient = 10                        # number of HOG orientations
    hog_pix_per_cell = 8                   # number of pixels per cell
    hog_cell_per_block = 2                 # number of cells per block
    hog_channel = 'ALL'                    # number of HOG channels, can be 0, 1, 2, or 'ALL'

    def __init__(self, model_filename=None):
        """ Initialization method.

        :param model_filename: Loads the trained classifier model and the scaler if not None.
        """

        if model_filename:
            self.load_model(model_filename)
        else:
            # initialize a linear SVC
            self.classifier = LinearSVC()

        self.cip = CoreImageProcessing()

    def load_training_dataset(self, path, max_samples=None):
        """ Loads and prepares the training dataset. All training images shall be PNG formatted.

        :param path:         Path to training images. The directory shall have the following structure.
                              - non-vehicles
                              - vehicles
        :param max_samples:  Number of max totally loaded samples (vehicle + non-vehicle samples). If None all samples
                             are loaded.

        :return: Returns the number loaded vehicles and non-vehicles images. Both are 0 if no image could be loaded.
        """

        search_path_vehicles = path.rstrip('/') + '/vehicles/*/*.png'
        search_path_non_vehicles = path.rstrip('/') + '/non-vehicles/*/*.png'

        images_vehicles = glob.glob(search_path_vehicles, recursive=True)
        images_non_vehicles = glob.glob(search_path_non_vehicles, recursive=True)

        if max_samples is not None:
            images_vehicles = random.sample(images_vehicles, max_samples)
            images_non_vehicles = random.sample(images_non_vehicles, max_samples)

        for img_path in images_vehicles:
            self.train_vehicles_path.append(img_path)

        for img_path in images_non_vehicles:
            self.train_non_vehicles_path.append(img_path)

        return len(self.train_vehicles_path), len(self.train_non_vehicles_path)

    def convert_color_space(self, img_rgb, color_space):
        """ Converts and RGB image to the specified color space.

        :param img_rgb:     Input RGB image.
        :param color_space: Target color space (RGB, HLV, LUV, HLS, YUV and YCrCb).

        :return: Returns the converted image. In case the color space is `RGB` the method returns a copy of the input
                 image.
        """

        if color_space != 'RGB':
            if color_space == 'HSV':
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            else:
                img = None
                print('ERROR: Not supported color space "{:s}."'.format(color_space), file=sys.stderr)
        else:
            img = np.copy(img_rgb)

        return img

    def extract_features(self, img_rgb):
        """ Returns the 1-dimensional feature vector of the input RGB image.

        :param img_rgb:         Input RGB image.

        :return: Returns a 1-dimensional feature vector of the specified color space.
        """

        # convert color space
        if self.color_space != 'RGB':
            img = self.convert_color_space(img_rgb, self.color_space)
        else:
            img = np.copy(img_rgb)

        # get features
        feature_set = []

        if self.spatial_binning_enabled:
            spatial_features = self.cip.bin_spatial(img, size=self.spatial_size)
            feature_set.append(spatial_features)

        if self.hist_channels_enabled:
            hist_features = self.cip.color_histogram(img, nb_bins=self.hist_nb_bins, features_vector_only=True)
            feature_set.append(hist_features)

        if self.hog_enabled:
            hog_features = self.cip.hog_features(img, channel=self.hog_channel, orient=self.hog_orient,
                                                 pix_per_cell=self.hog_pix_per_cell,
                                                 cell_per_block=self.hog_cell_per_block)
            feature_set.append(np.ravel(hog_features))

        return np.concatenate(feature_set)

    def train(self):
        """ Trains the classifier.

        Please ensure, that the training data has been loaded before  (see `load_training_dataset`).
        """

        vehicle_features = []
        non_vehicle_features = []

        print('Prepare vehicle features...', end='', flush=True)

        for img_path in self.train_vehicles_path:
            img = mpimg.imread(img_path)
            vehicle_features.append(self.extract_features(img))
        print('done')
        print('Prepare non-vehicle features...', end='', flush=True)

        for img_path in self.train_non_vehicles_path:
            img = mpimg.imread(img_path)
            non_vehicle_features.append(self.extract_features(img))
        print('done')
        print('Normalize training features...', end='', flush=True)

        X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)

        # normalize training samples
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)

        # define the labels vector
        y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

        print('done')
        print('Split training features...', end='', flush=True)

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=rand_state)

        print('done')
        print()
        print('Configuration:')
        print('-----------------------------------------------')
        print(' Color space:             ', self.color_space)
        print(' Spatial binning enabled: ', self.spatial_binning_enabled)
        print(' Spatial binning:         ', self.spatial_size)
        print(' Histogram enabled:       ', self.hist_channels_enabled)
        print(' Number histogram bins:   ', self.hist_nb_bins)
        print(' HOG enabled:             ', self.hog_enabled)
        print(' HOG orientations:        ', self.hog_orient)
        print(' HOG pixel per cell:      ', self.hog_pix_per_cell)
        print(' HOG cells per block:     ', self.hog_cell_per_block)
        print(' HOG channel:             ', self.hog_channel)
        print(' Feature vector length:   ', len(X_train[0]))
        print()
        print('Train the classifier...', end='', flush=True)

        # train the classifier

        t_start = time.time()
        self.classifier.fit(X_train, y_train)
        t_end = time.time()

        print('done')
        print()
        print('Training Results:')
        print('-----------------------------------------------')
        print(' Training duration:      {:.3f} s'.format(t_end - t_start))
        print(' Train accuracy:         {:6.2f} %'.format(self.classifier.score(X_train, y_train) * 100.))
        print(' Test accuracy:          {:6.2f} %'.format(self.classifier.score(X_test, y_test) * 100.))

        # check the prediction time for a single sample
        t_start = time.time()
        n_predict = 1000
        self.classifier.predict(X_test[0:n_predict])
        t_end = time.time()
        print(' Prediction duration:    {:f} s'.format(t_end - t_start))

        return True

    def save_model(self, filename='model.pkl'):
        """ Saves the train classifier and the scaler in a pickle file.

        :param filename: Filename of the model.
        """
        with open(filename, 'wb') as f:
            pickle.dump((self.classifier, self.scaler), f)

    def load_model(self, filename='model.pkl'):
        """ Load the train classifier from a pickled file.

        :param filename: Filename of the trained model.
        """
        with open(filename, 'rb') as f:
            self.classifier, self.scaler = pickle.load(f)

    def predict(self, features):
        """ Classifies the features for vehicle or non-vehicle.

        :param features: Image (window) features to be classified.

        :return: Returns the classification (prediction) result and the confidence.
        """

        confidence = self.classifier.decision_function(features)
        prediction = self.classifier.predict(features)

        return prediction, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Classification')

    parser.add_argument(
        '-t', '--train',
        help='Trains the classifier with images in given input directory.',
        dest='train_directory',
        metavar='PATH'
    )
    parser.add_argument(
        '-s', '--save',
        help='Saves the trained classifier to model filename (*.pkl).',
        dest='model_filename',
        metavar='MODEL_FILE'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(0)

    if args.train_directory:
        # train the vehicle classifier with given samples

        vc = VehicleClassifier()

        print('Load training dataset...', end='', flush=True)
        # TODO: nb_train_vehicles, nb_train_non_vehicles = vc.load_training_dataset(args.train_directory, 2000)
        nb_train_vehicles, nb_train_non_vehicles = vc.load_training_dataset(args.train_directory)
        print('done')

        if nb_train_vehicles == 0 or nb_train_non_vehicles == 0:
            print('ERROR: Did not find all training (vehicle and non-vehicle) samples. Please check if the directories are setup correctly.', file=sys.stderr)
        else:
            nb_samples = nb_train_vehicles + nb_train_non_vehicles
            print('Dataset statistics:')
            print('  Total samples: {:d}'.format(nb_samples))
            print('  vehicles:      {:d} {:.2f} %'.format(nb_train_vehicles, nb_train_vehicles / nb_samples * 100.))
            print('  non-vehicles:  {:d} {:.2f} %'.format(nb_train_non_vehicles, nb_train_non_vehicles / nb_samples * 100.))

            vc.train()

            if args.model_filename:
                # save the trained classifier model
                print('Save trained classifier to {:s}...'.format(args.model_filename), end='', flush=True)
                vc.save_model(args.model_filename)
                print('done')
