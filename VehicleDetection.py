import argparse
import os
from CoreImageProcessing import *
from VehicleClassifier import *
from scipy.ndimage.measurements import label
import numpy as np
import threading
import click


class VehicleDetection:
    """ Detects and tracks vehicles in a series of images. """

    cip = None                             # Core image processing instance
    classifier = None                      # Vehicle classifier instance

    # -----------------------------------------------------------------------
    # Images of all processing steps for debug purposes

    fig_overlay = None                                            # Figure for debug plots
    debug_plots = []                                              # Array with axis (plots)
    img_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)            # input RGB image scaled 0 to 255
    img_rgb_scaled = np.zeros_like(img_rgb, dtype=np.float32)     # input RGB image scaled 0 to 1
    img_rgb_overlay = np.zeros_like(img_rgb, dtype=np.uint8)      # input RGB image with all overlays

    # -----------------------------------------------------------------------
    # Vehicle detection parameters

    image_format = 'jpg'                   # used image format ('jpg' scaled 0 to 255 and 'png' scaled 0 to 1)
    classifier_img_size = (64, 64)         # image size the classifier has been trained with
    y_start_stop = [400, 656]              # Min and max in y-direction to search in slide_window()

    roi_far_range = [400, 500, 1.1, 1]     # [ROI min y, ROI max y, scale factor, cells per step]
    roi_mid_range = [400, 600, 1.5, 2]     # [ROI min y, ROI max y, scale factor, cells per step]
    roi_near_range = [400, 656, 2.6, 2]    # [ROI min y, ROI max y, scale factor, cells per step]

    conf_threshold = 0.5                   # min confidence threshold for vehicle candidates
    history_nb_frames = 8                  # number of frames stored in the history arrays
    history_min_weight = 0.6               # min averaging weight for cumulated heatmap
    heatmap_threshold = 12.5               # Threshold for heatmap filtering
    heatmap_history = []                   # last n vehicle heatmaps
    heatmap = np.zeros_like(img_rgb[:, :, 0], dtype=np.float32)     # cumulated vehicle heatmap
    heatmap_thresholded = np.zeros_like(img_rgb[:, :, 0], dtype=np.float32)   # cumulated vehicle heatmap thresholded

    bboxes_candidates = []                 # Bounding boxes of all vehicle candidates (classified vehicles)
    bboxes_candidates_1 = []               # Bounding boxes of 1st scale factor vehicle candidates (classified vehicles)
    bboxes_candidates_2 = []               # Bounding boxes of 2nd scale factor vehicle candidates (classified vehicles)
    bboxes_candidates_3 = []               # Bounding boxes of 3rd scale factor vehicle candidates (classified vehicles)
    bboxes_confidence_1 = []               # Bounding boxes confidence of 1st scale factor vehicle candidates (classified vehicles)
    bboxes_confidence_2 = []               # Bounding boxes confidence of 2nd scale factor vehicle candidates (classified vehicles)
    bboxes_confidence_3 = []               # Bounding boxes confidence of 3rd scale factor vehicle candidates (classified vehicles)

    nb_vehicles = 0                        # Number of detected vehicles
    labels = []                            # Scipy heatmap labels

    # -----------------------------------------------------------------------
    # Overlay setup

    overlay_bboxes_candidates_enabled = True

    def __init__(self, model_filename='model.pkl'):
        """ Initialization method.

        :param model_filename: Loads the trained classifier model and the scaler.
        """

        self.cip = CoreImageProcessing()
        self.classifier = VehicleClassifier(model_filename)
        self.init_debug_plots()

    def init_debug_plots(self):
        """ Initialize figure for debugging plots. """

        # init main overlay image
        self.fig_overlay = plt.figure(figsize=(13, 7))
        self.fig_overlay.subplots_adjust(bottom=0.01, left=0.01, top=0.99, right=0.99, wspace=0.01, hspace=0.01)
        self.debug_plots.append(plt.imshow(self.img_rgb, vmin=0, vmax=255))
        plt.ion()
        plt.axis('off')

        # Definition of add_axes rect
        # rect = [left, bottom, width, height] in fractions of figure width and height

        # init heatmap plot
        ax = self.fig_overlay.add_axes([0.03, 0.73, 0.25, 0.25])
        ax.text(0.035, 0.85, 'Heatmap', verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, color='white', fontsize=10)
        self.debug_plots.append(ax.imshow(self.heatmap, cmap='jet', vmin=0, vmax=255))
        ax.axis('off')

        # init heatmap thresholded plot
        ax = self.fig_overlay.add_axes([0.275, 0.73, 0.25, 0.25])
        ax.text(0.035, 0.85, 'Heatmap Thresholded', verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, color='white', fontsize=10)
        self.debug_plots.append(ax.imshow(self.heatmap_thresholded, cmap='jet', vmin=0, vmax=255))
        ax.axis('off')

    def update_debug_plots(self):
        """ Updates the debug plots with latest pipeline results. """

        self.img_rgb_overlay = self.img_rgb.copy()

        if len(self.bboxes_candidates) > 0 and self.overlay_bboxes_candidates_enabled:
            self.draw_boxes(self.img_rgb_overlay, self.bboxes_candidates_1, self.bboxes_confidence_1, (255, 255, 0), 1)
            self.draw_boxes(self.img_rgb_overlay, self.bboxes_candidates_2, self.bboxes_confidence_2, (0, 255, 255), 1)
            self.draw_boxes(self.img_rgb_overlay, self.bboxes_candidates_3, self.bboxes_confidence_3, (255, 0, 255), 1)

        if len(self.labels) > 0:
            self.draw_labeled_bboxes(self.img_rgb_overlay, self.labels, (0, 255, 0), 4)

        self.debug_plots[0].set_data(self.img_rgb_overlay)
        self.debug_plots[1].set_data(self.heatmap * 10)
        self.debug_plots[2].set_data(self.heatmap_thresholded * 10)
        self.fig_overlay.canvas.draw()
        self.fig_overlay.canvas.flush_events()


    def draw_boxes(self, img_rgb, bboxes, confidence, color=(0, 0, 255), thickness=6):
        """ Draws bounding boxes on a copy of the input image.

        :param img_rgb:    Input RGB image.
        :param bboxes:     Array of bounding boxes. Each box is defined as [x0, y0, x1, y1].
        :param confidence: Confidence level for each bounding box.
        :param color:      Color (R, G, B) of the bounding box.
        :param thickness:  Thickness of the bounding box boarder.

        :return: Returns the input image with all bounding boxes.
        """

        for i, bbox in enumerate(bboxes):
            if len(bbox) > 0:
                cv2.rectangle(img_rgb, bbox[0], bbox[1], color, thickness)
                cv2.putText(img_rgb, '{:.3f}'.format(confidence[i]), (bbox[0][0], bbox[0][1]-3), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1)

        return img_rgb

    def draw_labeled_bboxes(self, img_rgb, labels, color=(0, 255, 0), thickness=6):
        """ Draws the labeled vehicle bounding boxes onto the RGB image.

        :param img_rgb:   Input RGB image.
        :param labels:    Scipy labels array.
        :param color:     Color (R, G, B) of the bounding box.
        :param thickness: Thickness of the bounding box boarder.

        :return:
        """

        for car_number in range(1, labels[1]+1):
            # find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            cv2.rectangle(img_rgb, bbox[0], bbox[1], (0, 255, 0), 6)

        return img_rgb

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None],
                     window_size=(64, 64), xy_overlap=(0.5, 0.5)):
        """ Creates a list sliding windows based on start/stop position, windows size and overlap specification.

        :param x_start_stop:  x start/stop position of sliding windows [x_start, x_stop]
        :param y_start_stop:  y start/stop position of sliding windows [y_start, y_stop]
        :param window_size:     Windows size (x_size, y_size)
        :param xy_overlap:    x and y window overlap fraction (x_overlap, y_overlap)

        :return: Returns a list of windows.
        """
        # if x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] is None:
            x_start_stop[0] = 0
        if x_start_stop[1] is None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] is None:
            y_start_stop[0] = 0
        if y_start_stop[1] is None:
            y_start_stop[1] = img.shape[0]

        # compute the span of the region to be searched
        x_span = x_start_stop[1] - x_start_stop[0]
        y_span = y_start_stop[1] - y_start_stop[0]

        # compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(window_size[0] * (1 - xy_overlap[0]))
        ny_pix_per_step = np.int(window_size[1] * (1 - xy_overlap[1]))

        # compute the number of windows in x/y
        nx_buffer = np.int(window_size[0] * (xy_overlap[0]))
        ny_buffer = np.int(window_size[1] * (xy_overlap[1]))
        nx_windows = np.int((x_span - nx_buffer) / nx_pix_per_step)
        ny_windows = np.int((y_span - ny_buffer) / ny_pix_per_step)

        window_list = []

        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # calculate window position
                start_x = xs * nx_pix_per_step + x_start_stop[0]
                end_x = start_x + window_size[0]
                start_y = ys * ny_pix_per_step + y_start_stop[0]
                end_y = start_y + window_size[1]
                window_list.append(((start_x, start_y), (end_x, end_y)))

        return window_list

    def search_windows(self, img, windows):
        """ Classifies all sliding windows and returns a list of vehicle candidates.

        :param windows: Array of sliding windows generated by the `slide_window()` method.

        :return: Returns an array of positive classified windows (vehicle candidates) .
        """

        # positive detection windows
        on_windows = []

        for window in windows:
            # extract test window from original image
            img_test = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], self.classifier_img_size)

            features = self.classifier.extract_features(img_test)

            # scale extracted features to be fed to classifier
            test_features = self.classifier.scaler.transform(np.array(features).reshape(1, -1))

            # predict using your classifier
            prediction = self.classifier.predict(test_features)

            if prediction == 1:
                on_windows.append(window)

        return on_windows

    def find_vehicles(self, img_rgb, y_start, y_stop, scale, cells_per_step):
        """ Applies the HOG feature calculation onto the entire ROI of the image to speed-up the feature extraction
        pipeline.

        :param img_rgb:         Input RGB image.
        :param y_start:         Min ROI y-value
        :param y_stop:          Max ROI y-value
        :param scale:           Scale factor defining the search window size.
        :param cells_per_step:  Number of cells to be stepped between two consecutive search windows.

        :return: Returns a list of bounding boxes classified as vehicles, its confidence, and the corresponding heatmap.
        """

        bboxes = []                                                   # vehicle bounding boxes (classified vehicles/candidates)
        confidence = []                                               # confidence of each bounding box
        heatmap = np.zeros_like(img_rgb[:, :, 0], dtype=np.float32)   # vehicle heatmap

        # crop image to region of interest (ROI)
        img_tosearch = img_rgb[y_start:y_stop, :, :]
        ctrans_tosearch = self.classifier.convert_color_space(img_tosearch, color_space=self.classifier.color_space)

        # scale ROI
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # define blocks and steps as above
        nxblocks = (ch1.shape[1] // self.classifier.hog_pix_per_cell) - self.classifier.hog_cell_per_block + 1
        nyblocks = (ch1.shape[0] // self.classifier.hog_pix_per_cell) - self.classifier.hog_cell_per_block + 1
        nfeat_per_block = self.classifier.hog_orient * self.classifier.hog_cell_per_block**2

        window = 64
        nblocks_per_window = window // self.classifier.hog_pix_per_cell - self.classifier.hog_cell_per_block + 1
        cells_per_step = cells_per_step  # instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # compute individual channel HOG features for the entire image
        hog1 = self.cip.hog_features_single_channel(ch1, self.classifier.hog_orient, self.classifier.hog_pix_per_cell, self.classifier.hog_cell_per_block, feature_vec=False)
        hog2 = self.cip.hog_features_single_channel(ch2, self.classifier.hog_orient, self.classifier.hog_pix_per_cell, self.classifier.hog_cell_per_block, feature_vec=False)
        hog3 = self.cip.hog_features_single_channel(ch3, self.classifier.hog_orient, self.classifier.hog_pix_per_cell, self.classifier.hog_cell_per_block, feature_vec=False)

        # Debug output of HOG feature images
        # hog4, hia = self.cip.hog_features(ctrans_tosearch, 'ALL', self.classifier.hog_orient, self.classifier.hog_pix_per_cell, self.classifier.hog_cell_per_block, feature_vec=False, vis=True)
        #
        # fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        # ax[0].imshow(img_tosearch)
        # ax[0].set_title('Search Image (ROI)')
        # ax[1].imshow(ctrans_tosearch)
        # ax[1].set_title('Scaled down and converted to YCrCb Color Space')
        # ax[2].imshow(hia, cmap='gray')
        # ax[2].set_title('HOG Orientations (All Channels)')
        # plt.tight_layout()
        # plt.draw()

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * self.classifier.hog_pix_per_cell
                ytop = ypos * self.classifier.hog_pix_per_cell

                # extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], self.classifier_img_size)

                # get color features
                spatial_features = self.cip.bin_spatial(subimg, size=self.classifier.spatial_size)
                hist_features = self.cip.color_histogram(subimg, nb_bins=self.classifier.hist_nb_bins)

                # scale features and make a prediction
                test_features = self.classifier.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction, test_confidence = self.classifier.predict(test_features)

                if test_prediction == 1 and test_confidence[0] >= self.conf_threshold:
                    # update bounding box, confidence and heatmap lists for
                    # vehicle canditates with min required confidence level
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    bboxes.append(((xbox_left, ytop_draw + y_start), (xbox_left + win_draw, ytop_draw + win_draw + y_start)))
                    confidence.append(test_confidence[0])
                    heatmap[ytop_draw + y_start:ytop_draw + win_draw + y_start, xbox_left:xbox_left + win_draw] += 1

        return bboxes, confidence, heatmap

    def detect_vehicles(self, img_rgb):
        """ Detects all vehicles.

        :param img_rgb:  Input RGB image (original image)

        :return: Returns the displayed RGB image with overlays.
        """

        self.img_rgb = img_rgb
        self.img_rgb_overlay = np.copy(img_rgb)

        if self.image_format == 'jpg':
            # classifier has been trained with png files (scaled 0 to 1)
            # when using jpg files (scaled 0 to 255) the scale has to be adjusted
            self.img_rgb_scaled = img_rgb.astype(np.float32) / 255
        else:
            self.img_rgb_scaled = img_rgb

        # search for vehicles candidates in near, mid and far range (HOG sub-sampling window search)
        bboxes_candidates_1, confidence_1, heatmap_1 = self.find_vehicles(self.img_rgb_scaled, self.roi_far_range[0], self.roi_far_range[1], self.roi_far_range[2], self.roi_far_range[3])
        bboxes_candidates_2, confidence_2, heatmap_2 = self.find_vehicles(self.img_rgb_scaled, self.roi_mid_range[0], self.roi_mid_range[1], self.roi_mid_range[2], self.roi_mid_range[3])
        bboxes_candidates_3, confidence_3, heatmap_3 = self.find_vehicles(self.img_rgb_scaled, self.roi_near_range[0], self.roi_near_range[1], self.roi_near_range[2], self.roi_near_range[3])

        heatmap = heatmap_1 + heatmap_2 + heatmap_3

        nb_bboxes_candidates_1 = np.array(bboxes_candidates_1).shape[0]
        nb_bboxes_candidates_2 = np.array(bboxes_candidates_2).shape[0]
        nb_bboxes_candidates_3 = np.array(bboxes_candidates_3).shape[0]
        self.bboxes_candidates = []
        self.bboxes_candidates_1 = []
        self.bboxes_candidates_2 = []
        self.bboxes_candidates_3 = []

        if nb_bboxes_candidates_1 > 0:
            for i, bbox in enumerate(bboxes_candidates_1):
                self.bboxes_candidates.append(bbox)
                self.bboxes_candidates_1.append(bbox)
                self.bboxes_confidence_1.append(confidence_1[i])
        if nb_bboxes_candidates_2 > 0:
            for i, bbox in enumerate(bboxes_candidates_2):
                self.bboxes_candidates.append(bbox)
                self.bboxes_candidates_2.append(bbox)
                self.bboxes_confidence_2.append(confidence_2[i])
        if nb_bboxes_candidates_3 > 0:
            for i, bbox in enumerate(bboxes_candidates_3):
                self.bboxes_candidates.append(bbox)
                self.bboxes_candidates_3.append(bbox)
                self.bboxes_confidence_3.append(confidence_3[i])

        # Debug output of heatmaps
        # img_debug = np.copy(img_rgb)
        #
        # if len(self.bboxes_candidates) > 0 and self.overlay_bboxes_candidates_enabled:
        #     self.draw_boxes(img_debug, self.bboxes_candidates_1, self.bboxes_confidence_1, (255, 255, 0), 1)
        #     self.draw_boxes(img_debug, self.bboxes_candidates_2, self.bboxes_confidence_2, (0, 255, 255), 1)
        #     self.draw_boxes(img_debug, self.bboxes_candidates_3, self.bboxes_confidence_3, (255, 0, 255), 1)
        #
        # fig = plt.figure(figsize=(17, 9))
        # ax = plt.subplot(2, 1, 1)
        # ax.imshow(img_debug)
        # ax.set_title('Bounding Boxes')
        # ax = plt.subplot(2, 4, 5)
        # ax.imshow(heatmap_1, cmap='jet')
        # ax.set_title('Heatmap FAR Range')
        # ax = plt.subplot(2, 4, 6)
        # ax.imshow(heatmap_2, cmap='jet')
        # ax.set_title('Heatmap MID Range')
        # ax = plt.subplot(2, 4, 7)
        # ax.imshow(heatmap_3, cmap='jet')
        # ax.set_title('Heatmap NEAR Range')
        # ax = plt.subplot(2, 4, 8)
        # ax.imshow(heatmap, cmap='jet')
        # ax.set_title('Combined Heatmap')
        # plt.tight_layout()
        # plt.draw()

        # check max number of history heatmaps
        if len(self.heatmap_history) >= self.history_nb_frames:
            self.heatmap_history.pop(0)

        # update heatmaps and cumulate history (old maps have lower weight)
        self.heatmap_history.append(heatmap)

        heatmap_cumulated = np.zeros_like(self.heatmap)
        dt_factor = self.history_min_weight / (self.history_nb_frames - 1)

        for i, map in enumerate(self.heatmap_history):
            heatmap_cumulated += map * (self.history_min_weight + (i * dt_factor))

        self.heatmap = heatmap_cumulated

        # threhold heatmap
        self.heatmap_thresholded = np.copy(self.heatmap)
        self.heatmap_thresholded[self.heatmap_thresholded <= self.heatmap_threshold] = 0

        # apply scipy label function to identify vehicle bounding box based on thresholded heatmap
        self.labels = label(np.clip(self.heatmap_thresholded, 0, 255)) # TODO: , structure=np.ones((3, 3), dtype="bool8"))
        self.nb_vehicles = self.labels[1] + 1

        self.update_debug_plots()

        # Debug output for heatmap history, thresholds and labeled data
        # self.cip.show_images(figsize=(17, 4), rows=1,
        #                      images=[self.img_rgb_overlay, self.heatmap, self.heatmap_thresholded, self.labels[0]],
        #                      titles=['Bounding Boxes', 'Cumulated History Heatmap', 'Thresholded Heatmap', 'Labels'],
        #                      cmaps=['', 'jet', 'jet', 'gray'])
        # plt.show()

        return self.img_rgb_overlay

def keyboard_thread():
    """ Keyboard input thread. """

    global running, start, end, idx, paused, step_one_frame, images

    while running:
        key = click.getchar()

        if key == 'q':
            running = False
            plt.close()
            print('Quit lane detection.')
        elif key == 'r':
            idx = start
            print('Restart lane detection ({:d}, {:d})'.format(start, end))
        elif key == '0':
            start = 0
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '1':
            start = 190
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '2':
            start = 460
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '3':
            start = 700
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '4':
            start = 750
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == '5':
            start = 900
            idx = start

            print('Selected start frame {:d}'.format(idx))
        elif key == '6':
            start = 1200
            idx = start
            print('Selected start frame {:d}'.format(idx))
        elif key == 's':
            filename = 'frame_{:04d}.jpg'.format(idx)
            print('Safe image {:s}...'.format(filename), end='', flush=True)
            cv2.imwrite(filename, cv2.cvtColor(images[idx], cv2.COLOR_RGB2BGR))
            print('done')
        elif key == 'p':
            if not paused:
                paused = not paused
                print('Paused at frame {:d}'.format(idx))
            else:
                paused = not paused
                print('Continued at frame {:d}'.format(idx))
        elif key == 'n':
            step_one_frame = True
            print('Step one frame {:d}'.format(idx))
        elif key == 'i':
            print('Frame index: {:d}'.format(idx))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Detection')

    parser.add_argument(
        '-i', '--input',
        help='Input video file.',
        dest='input_video',
        metavar='VIDEO_FILE'
    )
    parser.add_argument(
        '-o', '--output',
        help='Write images to given directory.',
        dest='output_dir',
        metavar='DIRECTORY'
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # no arguments found
        parser.print_usage()
        parser.exit(0)

    if args.output_dir:
        # check for valid output path
        if os.path.exists(args.output_dir):
            output_dir = args.output_dir
        else:
            print('ERROR: Output directory does not exists.')
            exit(-1)

    if args.input_video:
        # load video file
        if os.path.exists(args.input_video):
            print('Extracting images from video file...', end='', flush=True)
            images = CoreImageProcessing.load_video(args.input_video)
            print('done')
            print('Number of images: {:d}'.format(len(images)))
        else:
            print('ERROR: Video file {:s} not found.'.format(args.input_video))
            exit(-1)

    # run vehicle detection processing pipeling
    vd = VehicleDetection(model_filename='model.pkl')

    click_enabled = True
    running = True

    start = 0
    end = len(images) - 1
    idx = start
    paused = False
    step_one_frame = False

    if click_enabled:
        thread = threading.Thread(target=keyboard_thread)
        thread.start()
        print('Started keyboard thread')

    while running:
        if idx > end:
            print('Last frames processed. Restart application.')
            break

        if not paused or step_one_frame:
            img_vd = vd.detect_vehicles(images[idx])
            idx += 1
            step_one_frame = False

        if args.output_dir:
            filename = output_dir + '/frame_{:04d}.png'.format(idx)
            print('Writing image {:s}'.format(filename))
            vd.fig_overlay.savefig(filename)

        plt.pause(0.0000001)

    if click_enabled:
        thread.join()

    plt.show()
