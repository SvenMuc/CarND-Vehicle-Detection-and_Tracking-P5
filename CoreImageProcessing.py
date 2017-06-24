import argparse
import numpy as np
import cv2
import sys
import matplotlib
#matplotlib.use('macosx', force=True)  # does not supports all features on macos environments
matplotlib.use('TKAgg', force=True)   # slow but stable on macosx environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import VideoFileClip
from skimage.feature import hog


class CoreImageProcessing:
    """ Provides core image processing methods likes gradient calculation, gradient direction, magnitude
    thresholds, etc. """

    def __init__(self):
        """ Initialization method. """

    @staticmethod
    def show_image(image, title='', cmap=None, axis='off', show=False):
        """ Show a single image in a matplotlib figure.

        :param image:  Image to be shown.
        :param title:  Image title.
        :param cmap:   Colormap (most relevant: 'gray', 'jet', 'hsv')
                       For supported colormaps see: https://matplotlib.org/examples/color/colormaps_reference.html
        :param axis:   Activates 'on' or deactivates 'off' the x- and y-axis.
        :param show:   If true, the image will be shown immediately. Otherwise `plt.show()` shall be called at a later
                       stage.
        """

        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis(axis)

        if show:
            plt.show()

    @staticmethod
    def show_images(figsize, rows, images, titles=[], cmaps=[], fig_title='', axis='off', show=False):
        """ Show a single image in a matplotlib figure.

        :param figsize:   Size of the image in inch (width, height).
        :param rows:      Number of rows.
        :param images:    1D-Array of images to be shown.
        :param titles:    1D-Array of image titles.
        :param cmaps:     1D-Array of colormaps (most relevant: 'gray', 'jet', 'hsv'). Use '' to apply default cmap.
                          For supported colormaps see: https://matplotlib.org/examples/color/colormaps_reference.html
        :param fig_title: Figure title.
        :param axis:      Activates 'on' or deactivates 'off' the x- and y-axis.
        :param show:      If true, the image will be shown immediately. Otherwise `plt.show()` shall be called at a
                          later stage.
        """

        nb_images = len(images)
        nb_images_per_row = int(nb_images / rows)

        fig, axarr = plt.subplots(rows, nb_images_per_row, figsize=figsize)
        fig.tight_layout()

        if fig_title != '':
            fig.suptitle(fig_title)
            plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.01, hspace=0.15)

        if rows == 1 or nb_images_per_row == 1:
            # plot single row
            for i, ax in enumerate(axarr):
                if cmaps[i] == '':
                    ax.imshow(images[i])
                else:
                    ax.imshow(images[i], cmap=cmaps[i])

                ax.set_title(titles[i])
                ax.axis(axis)
        else:
            # plot multiple rows
            idx = 0
            for r in range(rows):
                for c in range(nb_images_per_row):
                    if cmaps[idx] == '':
                        axarr[r][c].imshow(images[idx])
                    else:
                        axarr[r][c].imshow(images[idx], cmap=cmaps[idx])

                    axarr[r][c].set_title(titles[idx])
                    axarr[r][c].axis(axis)
                    idx += 1

        if show:
            plt.show()

    @staticmethod
    def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
        """ Plot pixels in 3D.

        :param pixels:       3D pixel coordinates.
        :param colors_rgb:   Scales RGB colors [0..1].
        :param axis_labels:  List of x-, y- and z-labels.
        :param axis_limits:  Array defining the x-, y- and z-axis limits
                             [(x_min, x_max), (y_min, y_max), (z_min, z_max)]

        :return: Returns an Axes3D object for further manipulation.
        """

        # Create figure and 3D axes
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)

        # Set axis limits
        ax.set_xlim(*axis_limits[0])
        ax.set_ylim(*axis_limits[1])
        ax.set_zlim(*axis_limits[2])

        # Set axis labels and sizes
        ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
        ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
        ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
        ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

        # Plot pixel values with colors given in colors_rgb
        ax.scatter(
            pixels[:, :, 0].ravel(),
            pixels[:, :, 1].ravel(),
            pixels[:, :, 2].ravel(),
            c=colors_rgb.reshape((-1, 3)), edgecolors='none')

        # return Axes3D object for further manipulation
        return ax

    def threshold_image_channel(self, image_channel, threshold=(0, 255)):
        """ Thresholds a single image channel.

        :param image_channel: Single image channel (e.g. R, G, B or H, L, S channel)
        :param threshold:    Min/max color thresholds [0..255].

        :return: Returns a thresholded binary image.
        """

        binary = np.zeros_like(image_channel, dtype=np.uint8)
        binary[(image_channel >= threshold[0]) & (image_channel <= threshold[1])] = 1

        return binary

    def gradient(self, image, sobel_kernel=3, orientation='x'):
        """ Calculates the gradient of the image channel in x-, y- or in x- and y-direction.

        :param image:        Single channel image.
        :param sobel_kernel: Sobel kernel size. Min 3.
        :param orientation:  Gradient orientation ('x' = x-gradient, 'y' = y-gradient)

        :return: Returns the gradient or None in case of an error.
        """

        if orientation == 'x':
            return cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orientation == 'y' or orientation == 'xy':
            return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            print('ERROR: Not supported gradient orientation (x or y supported only).', file=sys.stderr)
            return None

    def norm_abs_gradient(self, gradient):
        """ Calculates the normalized absolute directional gradients.

        :param gradient:     x- or y-gradients

        :return: Returns normalized [0..255] absolute gradient.
        """

        abs_gradient = np.absolute(gradient)
        return np.uint8(255 * abs_gradient / np.max(abs_gradient))

    def abs_gradient_threshold(self, gradient, threshold=(0, 255)):
        """ Calculates the absolute directional gradients and applies a threshold.

        :param gradient:     x- or y-gradients
        :param orientation:  Gradient orientation used for debug plots only.
                             ('' = no title, 'x' = x-gradient title, 'y' = y-gradient title)
        :param threshold:    Min/max thresholds of gradient [0..255].

        :return: Returns a thresholded gradient binary image.
        """

        abs_gradient = self.norm_abs_gradient(gradient)
        binary = np.zeros_like(abs_gradient)
        binary[(abs_gradient >= threshold[0]) & (abs_gradient <= threshold[1])] = 1

        return binary

    def norm_magnitude(self, gradient_x, gradient_y):
        """ Calculates the normalized magnitude of the x- and y-gradients.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients

        :return: Returns a normalized [0..255] magnitude.
        """

        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        return np.uint8(255 * magnitude / np.max(magnitude))

    def magnitude_threshold(self, gradient_x, gradient_y, threshold=(0, 255)):
        """ Calculates the magnitude of the x- and y-gradients and applies a threshold.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients
        :param threshold:   Min/max thresholds of magnitude [0..255].

        :return: Returns a thresholded magnitude binary image.
        """

        magnitude = self.norm_magnitude(gradient_x, gradient_y)
        binary = np.zeros_like(magnitude)
        binary[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1

        return binary

    def direction(self, gradient_x, gradient_y):
        """ Calculates the direction of the x- and y-gradients.

        :param gradient_x:  x-gradients
        :param gradient_y:  y-gradients

        :return: Returns the gradients' direction (angles).
        """

        return np.arctan2(gradient_y, gradient_x)

    def direction_threshold(self, gradient_x, gradient_y, threshold_pos=(0, np.pi/2), threshold_neg=(0, -np.pi/2)):
        """ Calculates the direction of the x- and y-gradients and applies a threshold.

        :param gradient_x:     x-gradients
        :param gradient_y:     y-gradients
        :param threshold_pos:  Positive min/max thresholds of direction [0..PI/2].
        :param threshold_neg:  Negative min/max thresholds of direction [0..PI/2].

        :return: Returns a thresholded direction binary image.
        """

        angles = self.direction(gradient_x, gradient_y)
        binary = np.zeros_like(angles)
        binary[(angles >= threshold_pos[0]) & (angles <= threshold_pos[1])] = 1
        binary[(angles <= threshold_neg[0]) & (angles >= threshold_neg[1])] = 1

        return binary

    def warp(self, image, src_pts, dst_pts, dst_img_size=None):
        """ Warps an image from source points to the destination points.

        :param image:        Input image.
        :param src_pts:      Source points (at least 4 points required).
        :param dst_pts:      Destination points (at least 4 points required).
        :param dst_img_size: Size of destination image (width, height). If None, use source image size.

        :return: Returns the warp image.
        """

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if dst_img_size is None:
            dst_img_size = (image.shape[1], image.shape[0])

        return cv2.warpPerspective(image, M, dst_img_size, flags=cv2.INTER_LINEAR)

    def histogram(self, image, roi=None):
        """ Calculates the histogram of an image channel.

        :param image: Input image (single channel).
        :param roi:   Image region in which the histogram should be calculated. If None the image is used.
                      Format: [x1, x2, y1, y2]

        :return: Returns the histogram.
        """

        if roi is not None:
            mask = np.zeros(image.shape[:2], np.uint8)
            mask[roi[2]:roi[3], roi[0]:roi[1]] = 255
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            hist = cv2.calcHist([image], [0], mask, [256], [0, 256])
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        return hist

    def color_histogram(self, image, nb_bins=32, features_vector_only=True):
        """ Calculates the histogram for each individual channel.

        :param image:       3 channel input image (e.g. RGB, HSV, YUV, LUV, etc.)
        :param nb_bin:      Number of bins.
        :param features_vector_only: If true the function only returns the histogram feature set.

        :return: Return the individual channel histograms, bin_centers and feature vectors. If
                 `hist_features_only == True`the function returns the feature vector only
        """

        hist_c0 = np.histogram(image[:, :, 0], bins=nb_bins)
        hist_c1 = np.histogram(image[:, :, 1], bins=nb_bins)
        hist_c2 = np.histogram(image[:, :, 2], bins=nb_bins)

        # generate bin centers
        bin_edges = hist_c0[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # concatenate the histograms into a single feature vector
        hist_features = np.concatenate((hist_c0[0], hist_c1[0], hist_c2[0]))

        if features_vector_only:
            return hist_features
        else:
            return hist_c0, hist_c1, hist_c2, bin_centers, hist_features

    def bin_spatial(self, img, size=(32, 32)):
        """ Creates a 1-dimensional feature vector of an spatially binned 3-channel image.

        :param img:         3 channel input image.
        :param size:        Spatial image size applied for the feature vector.

        :return: Returns the 1-dimensional spatial image feature vector.
        """

        img_feature = np.copy(img)

        # create the 1-dimensional feature vector
        return cv2.resize(img_feature, size).ravel()

    def hog_features_single_channel(self, img_channel, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        """ Calculates the HOG (Histogram of Oriented Gradients) features and the HOG image if `vis` is set to True on
        a single channel image.

        Further details about the algo can be found here:
          http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

        :param img_channel:     Single channel input image
        :param orient:          Number of orientation bins.
        :param pix_per_cell:    Size of a cell in pixels.
        :param cell_per_block:  Number of cells in each block.
        :param vis:             If true return the HOG image.
        :param feature_vec:     Return the data as a feature vector by calling .ravel() on the result just
                                before returning.

        :return: Returns the HOG features and if `vis=True` the HOG features and the HOG image.
        """

        if vis:
            # return hog features and hog image
            features, img_hog = hog(img_channel, orientations=orient,
                                    pixels_per_cell=(pix_per_cell, pix_per_cell),
                                    cells_per_block=(cell_per_block, cell_per_block),
                                    transform_sqrt=False,
                                    visualise=True,
                                    feature_vector=False)
            return features, img_hog
        else:
            # return hog features only
            features = hog(img_channel, orientations=orient,
                           pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block),
                           transform_sqrt=False,
                           visualise=False,
                           feature_vector=feature_vec)
            return features

    def hog_features(self, img, channel, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
        """ Calculates the HOG (Histogram of Oriented Gradients) features and the HOG image in `vis` is set to True.

        Further details about the algo can be found here:
          http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

        :param img:             Input image
        :param channel:         Image channel. Can be 0, 1, 2 or 'ALL'.
        :param orient:          Number of orientation bins.
        :param pix_per_cell:    Size of a cell in pixels.
        :param cell_per_block:  Number of cells in each block.
        :param vis:             If true return the HOG image.
        :param feature_vec:     Return the data as a feature vector by calling .ravel() on the result just
                                before returning.

        :return: Returns the HOG features and if `vis=True` the HOG features and the HOG image.
        """

        if vis:
            # return hog features and hog image
            if channel == 'ALL':
                features = []
                imgs_hog = []
                for ch in range(img.shape[2]):
                    feature, img_hog = self.hog_features_single_channel(img[:, :, ch], orient=orient, pix_per_cell=pix_per_cell,
                                                                        cell_per_block=cell_per_block, vis=True, feature_vec=True)
                    features.append(feature)
                    imgs_hog.append(img_hog)
            else:
                features, imgs_hog = self.hog_features_single_channel(img[:, :, channel], orient=orient, pix_per_cell=pix_per_cell,
                                                            cell_per_block=cell_per_block, vis=True, feature_vec=True)
            return features, img_hog
        else:
            # return hog features only
            if channel == 'ALL':
                features = []
                for ch in range(img.shape[2]):
                    features.append((self.hog_features_single_channel(img[:, :, ch], orient=orient, pix_per_cell=pix_per_cell,
                                                                      cell_per_block=cell_per_block, vis=False, feature_vec=True)))
            else:
                features = self.hog_features_single_channel(img[:, :, channel], orient=orient, pix_per_cell=pix_per_cell,
                                                            cell_per_block=cell_per_block, vis=False, feature_vec=True)
            return features

    @staticmethod
    def load_video(filename):
        """ Load video file and extract images.

        :param filename: mp4 filename

        :return: Returns extracted RGB images.
        """

        clip = VideoFileClip(filename)
        images = []

        for frame in clip.iter_frames():
            images.append(frame)

        return images

#
# Test environment for Core Image Processing class
#
def test_warp(img_rgb):
    """ Test image warping.

    :param img_rgb: Input RGB image.
    """

    # undistort image
    img_rgb = calib.undistort(img_rgb)

    # point order = btm left --> btm right --> btm left --> btm right
    height = img_rgb.shape[0]
    src_pts = np.float32([[193, height], [1117, height], [686, 450], [594, 450]])
    dst_pts = np.float32([[300, height], [977, height], [977, 0], [300, 0]])

    img_warped = cip.warp(img_rgb, src_pts, dst_pts)
    cv2.polylines(img_rgb, np.int32([src_pts]), isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(img_warped, np.int32([dst_pts]), isClosed=True, color=(255, 0, 0), thickness=2)

    cip.show_images(figsize=(10, 4), rows=1,
                    images=[img_rgb, img_warped],
                    titles=['Original Image', 'Warped Image'],
                    cmaps=['', ''])
    plt.show()


def test_color_space_3d_plots(img_rgb):
    """ Analyses color spaces for vehicle and non-vehicle images.

    :param img_rgb: Input RGB image.
    """
    # select a small fraction of pixels to plot by subsampling it
    scale = max(img_rgb.shape[0], img_rgb.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img_rgb, (np.int(img_rgb.shape[1] / scale), np.int(img_rgb.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)

    # convert subsampled image to desired color space(s)
    img_small_RGB = img_small
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

    # plot and show
    CoreImageProcessing.show_images(figsize=(7, 3), rows=1,
                                    images=[img_small_RGB, img_small_HSV, img_small_LUV],
                                    titles=['RGB', 'HSV', 'LUV'],
                                    cmaps=['', '', ''])

    CoreImageProcessing.plot3d(img_small_RGB, img_small_rgb)
    CoreImageProcessing.plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    CoreImageProcessing.plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
    plt.show()


def test_color_histogram(img_rgb):
    """ Test the color histogram function on an RGB image.

    :param img_rgb:  Input RGB image.
    """

    cip = CoreImageProcessing()
    rh, gh, bh, bincen, feature_vec = cip.color_histogram(img_rgb, nb_bins=32, bins_range=(0, 256), features_vector_only=False)

    # Plot a figure with all three bar charts
    if rh is not None and gh is not None and bh is not None:
        CoreImageProcessing.show_image(img_rgb)
        fig = plt.figure(figsize=(12,3))
        plt.subplot(131)
        plt.bar(bincen, rh[0])
        plt.xlim(0, 256)
        plt.title('R Histogram')
        plt.subplot(132)
        plt.bar(bincen, gh[0])
        plt.xlim(0, 256)
        plt.title('G Histogram')
        plt.subplot(133)
        plt.bar(bincen, bh[0])
        plt.xlim(0, 256)
        plt.title('B Histogram')
        fig.tight_layout()
    else:
        print('ERROR: color_histogram() returned None for at least one variable.', file=sys.stderr)


def test_bin_spatial(img_rgb):
    """ Test the bin spatial function on an RGB image.

    :param img_rgb:  Input RGB image.
    """
    cip = CoreImageProcessing()

    # convert RGB image to new color space
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

    features_rgb = cip.bin_spatial(img_rgb, size=(32, 32))
    features_hsv = cip.bin_spatial(img_hsv, size=(32, 32))
    features_luv = cip.bin_spatial(img_luv, size=(32, 32))
    features_hls = cip.bin_spatial(img_hls, size=(32, 32))
    features_yuv = cip.bin_spatial(img_yuv, size=(32, 32))
    features_ycrcb = cip.bin_spatial(img_ycrcb, size=(32, 32))

    fig, axarr = plt.subplots(2, 6, figsize=(15, 6))
    plt.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.05, wspace=0.2, hspace=0.2)
    axarr[0][0].imshow(img_rgb)
    axarr[0][1].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV))
    axarr[0][2].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV))
    axarr[0][3].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS))
    axarr[0][4].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV))
    axarr[0][5].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb))
    axarr[1][0].plot(features_rgb)
    axarr[1][1].plot(features_hsv)
    axarr[1][2].plot(features_luv)
    axarr[1][3].plot(features_hls)
    axarr[1][4].plot(features_yuv)
    axarr[1][5].plot(features_ycrcb)
    axarr[0][0].set_title('RGB')
    axarr[0][1].set_title('HSV')
    axarr[0][2].set_title('LUV')
    axarr[0][3].set_title('HLS')
    axarr[0][4].set_title('YUV')
    axarr[0][5].set_title('YCrCb')


def test_hog(img_rgb):
    """ Test the HOG feature function on an RGB image.

    :param img_rgb:  Input RGB image.
    """

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2

    cip = CoreImageProcessing()
    features, img_hog = cip.hog_features(img_gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

    fig, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_gray, cmap='gray')
    axarr[0].set_title('Original Image')
    axarr[1].imshow(img_hog, cmap='gray')
    axarr[1].set_title('HOG Visualization')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIP - Core Image Processing')

    parser.add_argument(
        '-tcs', '--test-color-spaces',
        help='Tests different color spaces in a 3D plot.',
        action='store_true'
    )
    parser.add_argument(
        '-tch', '--test-color-histogram',
        help='Tests color histogram on RGB image.',
        action='store_true'
    )
    parser.add_argument(
        '-tbs', '--test-bin-spatial',
        help='Tests bin spacial on RGB image.',
        action='store_true'
    )
    parser.add_argument(
        '-th', '--test-hog',
        help='Tests HOG features on grayscale image.',
        action='store_true'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(1)

    print('-----------------------------------------------------------------------------')
    print(' CIP - Core Image Processing Tests')
    print('-----------------------------------------------------------------------------')

    # configure core image processing
    cip = CoreImageProcessing()

    img_files = ['test_images/test1.jpg',
                 'test_images/test2.jpg',
                 'test_images/test3.jpg',
                 'test_images/test4.jpg',
                 'test_images/test5.jpg',
                 'test_images/test6.jpg',
                 'test_images/vehicle_25.png',
                 'test_images/vehicle_31.png',
                 'test_images/vehicle_53.png',
                 'test_images/non-vehicle_2.png',
                 'test_images/non-vehicle_3.png',
                 'test_images/non-vehicle_8.png']

    img_rgb = []

    for f in img_files:
        print('Load image file: {:s}'.format(f))
        img_rgb.append(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))

    if args.test_color_spaces:
        # test color spaces of vehicle and non-vehicle images in a 3D plot
        test_color_space_3d_plots(img_rgb[6])       # vehicle image
        test_color_space_3d_plots(img_rgb[7])       # vehicle image
        test_color_space_3d_plots(img_rgb[8])       # vehicle image
        test_color_space_3d_plots(img_rgb[9])       # non-vehicle image
        test_color_space_3d_plots(img_rgb[10])      # non-vehicle image
        test_color_space_3d_plots(img_rgb[11])      # non-vehicle image

    if args.test_color_histogram:
        # test color histogram on RGB images
        test_color_histogram(img_rgb[7])            # vehicle image
        test_color_histogram(img_rgb[9])            # non-vehicle image
        plt.show()

    if args.test_bin_spatial:
        # test spatial binning of different color spaces
        test_bin_spatial(img_rgb[6])               # vehicle image
        test_bin_spatial(img_rgb[7])               # vehicle image
        test_bin_spatial(img_rgb[9])               # non-vehicle image
        plt.show()

    if args.test_hog:
        # test HOG features on a grayscale image
        test_hog(img_rgb[6])                       # vehicle image
        test_hog(img_rgb[7])                       # vehicle image
        test_hog(img_rgb[8])                       # vehicle image
        test_hog(img_rgb[9])                       # non-vehicle image
        test_hog(img_rgb[10])                      # non-vehicle image
        test_hog(img_rgb[11])                      # non-vehicle image
        plt.show()

    # -----------------------------------------------------------------------
    # Test and optimize single images
    #test_preprocessing_pipeline(img_rgb[0], plot_intermediate_results=True)    # best case (black ground yellow/white)
    #test_preprocessing_pipeline(img_rgb[1], plot_intermediate_results=True)    # best case (black ground white)
    #test_preprocessing_pipeline_for_lane_detection(img_rgb[2], plot_intermediate_results=True)    # critical for R/B channel threshold
    #test_preprocessing_pipeline(img_rgb[3], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[4], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[5], plot_intermediate_results=True)    # critical for R channel threshold
    #test_preprocessing_pipeline(img_rgb[6], plot_intermediate_results=True)    # critical for R/S channel threshold
    #test_preprocessing_pipeline(img_rgb[8], plot_intermediate_results=True)    # critical for R channel threshold
    #test_preprocessing_pipeline(img_rgb[9], plot_intermediate_results=True)    # critical for R channel threshold
    #test_preprocessing_pipeline(img_rgb[10], plot_intermediate_results=True)   # critical for R channel threshold
    #test_preprocessing_pipeline(img_rgb[11], plot_intermediate_results=True)
    #test_preprocessing_pipeline(img_rgb[12], plot_intermediate_results=True)   # critical for S channel threshold
    #test_preprocessing_pipeline(img_rgb[13], plot_intermediate_results=True)   # critical for S channel threshold
    #test_preprocessing_pipeline(img_rgb[14], plot_intermediate_results=True)   # critical for S/B channel threshold
    #plt.show()
    #exit(0)

    # -----------------------------------------------------------------------
    # test warping
    #test_warp(img_rgb[0])
    #exit(0)

    # -----------------------------------------------------------------------
    # Pre-process all test images
    # img_preprocessed = []
    # titles = []
    # cmaps = []
    #
    # for i, img in enumerate(img_rgb):
    #     print('Pre-process image {:s}'. format(img_files[i]))
    #     img_preprocessed.append(img)
    #     img_preprocessed.append(test_preprocessing_pipeline_for_lane_detection(img))
    #     titles.extend([img_files[i], '(R&S)&B Binary Image'])
    #     cmaps.extend(['', 'gray'])
    #
    # cip.show_images(figsize=(12, 9), rows=4, fig_title='Pre-processing Results',
    #                 images=img_preprocessed,
    #                 titles=titles,
    #                 cmaps=cmaps)
    #
    # plt.draw()
    # plt.pause(1e-3)
    # plt.show()
