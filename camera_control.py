import cv2
import numpy as np
import threading
import time
import ipywidgets as widgets
from IPython.display import display
from PIL import Image
from torchvision import transforms
import os
import pandas as pd

class CameraControl:
    def __init__(self, digit_cam_id=0, color_cam_id=1, SCREEN_HEIGHT=680, SCREEN_WIDTH=840):
        self.digit_cam_id = digit_cam_id
        self.color_cam_id = color_cam_id
        
        self.digit_cam = None
        self.color_cam = None
        
        self.is_running = False
        self.thread = None

        self._video_writer = None
        self._color_video_writer = None
        self._recording = False
        self._frame_hw = None
        self._color_frame_hw = None
        self._record_start_time = None
        self._hsv_log_file = None
        
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        # 图片保存目录
        self._save_dir = "images"
        os.makedirs(f"{self._save_dir}/exact", exist_ok=True)
        os.makedirs(f"{self._save_dir}/nn_input", exist_ok=True)

        # ROI for digit camera
        self.digit_left_top = (200, 200)
        self.digit_right_bot = (350, 350)

        # ROI for color camera
        self.color_left_top = (200, 200)
        self.color_right_bot = (350, 350)

        # red dot detection
        self.center_x_threshold = 50
        self.dot_in_center = False

        # color detection
        self.lower_hsv = [0, 0, 0]
        self.upper_hsv = [360, 100, 100]
        self.min_pixels_ratio = 0.6

        # Neural network input image
        self.nn_input_img = None

    def open_camera(self):
        """Open both digit and color cameras; return True if successful."""
        if self.digit_cam is None or not self.digit_cam.isOpened():
            self.digit_cam = cv2.VideoCapture(self.digit_cam_id)
            if not self.digit_cam.isOpened():
                print(f"Unable to open digit camera (ID={self.digit_cam_id}).")
                return False

        if self.color_cam is None or not self.color_cam.isOpened():
            self.color_cam = cv2.VideoCapture(self.color_cam_id)
            if not self.color_cam.isOpened():
                print(f"Unable to open color camera (ID={self.color_cam_id}).")
                return False

        ret, test_frame = self.digit_cam.read()
        if ret and test_frame is not None:
            self._frame_hw = (test_frame.shape[0], test_frame.shape[1])

        ret_color, test_color_frame = self.color_cam.read()
        if ret_color and test_color_frame is not None:
            self._color_frame_hw = (test_color_frame.shape[0], test_color_frame.shape[1])

        return True

    def close_camera(self):
        """Close both cameras and destroy OpenCV windows."""
        self.stop_recording()
        if self.digit_cam:
            self.digit_cam.release()
            self.digit_cam = None

        if self.color_cam:
            self.color_cam.release()
            self.color_cam = None

        cv2.destroyAllWindows()
        self.is_running = False

    def start_recording(self, filename):
        if self._recording:
            return
        if self._frame_hw is None:
            print("Frame size not available, cannot start recording")
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        h, w = self._frame_hw
        self._video_writer = cv2.VideoWriter(filename, fourcc, 30, (w, h))

        if self._color_frame_hw is not None:
            ch, cw = self._color_frame_hw
            color_filename = filename.replace('.avi', '_color.avi')
            self._color_video_writer = cv2.VideoWriter(color_filename, fourcc, 30, (cw, ch))

        # Open HSV log Excel
        hsv_filename = filename.replace('.avi', '_hsv.xlsx')
        self._hsv_data = []
        self._hsv_filename = hsv_filename

        self._recording = True
        self._record_start_time = time.time()

    def stop_recording(self):
        self._recording = False
        self._record_start_time = None
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        if self._color_video_writer:
            self._color_video_writer.release()
            self._color_video_writer = None
        if hasattr(self, '_hsv_data') and self._hsv_data:
            df = pd.DataFrame(self._hsv_data, columns=['time_s', 'H', 'S', 'V'])
            df.to_excel(self._hsv_filename, index=False)
            self._hsv_data = None
            
    def process_digit_img(self, frame):
        """Process one frame from the digit camera."""
        ret = True
        frame_boxed, roi = draw_box_extract(ret, frame, self.digit_left_top, self.digit_right_bot)
        
        if self._recording and self._video_writer is not None:
            self._video_writer.write(frame_boxed)

        # White image for fallback
        white = np.ones((roi.shape[0], roi.shape[1], 3), dtype=np.uint8) * 255
        
        # Detect red dot in ROI
        red_dot_x, red_dot_y, contours = detect_red_dot(roi)
        
        # Calculate the center of the ROI (x-axis)
        roi_center_x = (self.digit_right_bot[0] + self.digit_left_top[0]) // 2

        # Draw red dot
        frame_boxed = draw_red_dot(frame_boxed, red_dot_x, red_dot_y, contours,
                                   self.digit_left_top, self.digit_right_bot)
        dot_in_center = False
        if red_dot_x is not None:
            global_x = red_dot_x + self.digit_left_top[0]  # Convert to global coords
            if abs(global_x - roi_center_x) < self.center_x_threshold:
                cv2.putText(frame_boxed, "Red dot centered!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                dot_in_center = True

        # Extract digit (must happen BEFORE drawing threshold lines,
        # because roi is a view into frame and drawing would corrupt it)
        extracted_img, extracted_PIL_img = extract_number_img(roi)

        # Draw center-threshold vertical lines (after extraction so they don't affect ROI)
        line_left_x  = roi_center_x - self.center_x_threshold
        line_right_x = roi_center_x + self.center_x_threshold
        y_top = self.digit_left_top[1]
        y_bot = self.digit_right_bot[1]
        cv2.line(frame_boxed, (line_left_x,  y_top), (line_left_x,  y_bot), (255, 200, 0), 1)
        cv2.line(frame_boxed, (line_right_x, y_top), (line_right_x, y_bot), (255, 200, 0), 1)
        nn_input_img = None
        if extracted_img is None:
            extracted_img = white
            processed_img_resized = white
        else:
            _, processed_img_resized, nn_input_img = preprocess_to_MNIST(extracted_PIL_img)
            
            # 当检测到数字图像 且 卡片位置正确 且正在录制时保存
            if dot_in_center and self._recording and self._record_start_time is not None:
                elapsed_ms = int((time.time() - self._record_start_time) * 1000)
                # 保存 extracted_img
                cv2.imwrite(
                    f"{self._save_dir}/exact/exact_{elapsed_ms}.png",
                    extracted_img
                )
                # 保存 nn_input_img（反归一化到 0-255 整数）
                if nn_input_img is not None:
                    with open(f"{self._save_dir}/nn_input/nn_{elapsed_ms}.txt", 'w') as f:
                        nn_input_to_save = (nn_input_img * 255).astype(np.uint8)
                        np.savetxt(f, nn_input_to_save, delimiter=',', fmt='%d')

        return frame_boxed, extracted_img, processed_img_resized, nn_input_img, dot_in_center

    def process_color_img(self, frame):
        def common_hsv_to_opencv(h_common, s_common, v_common):
            h_opencv = round(h_common / 2.0)
            s_opencv = round(s_common * 2.55)
            v_opencv = round(v_common * 2.55)
            hsv_opencv = np.array([h_opencv, s_opencv, v_opencv], dtype=np.uint8)
            return hsv_opencv
        lower_hsv_cv = common_hsv_to_opencv(*self.lower_hsv)
        upper_hsv_cv = common_hsv_to_opencv(*self.upper_hsv)

        """Simple color-based check."""
        cv2.rectangle(frame, self.color_left_top, self.color_right_bot, (0, 255, 0), 2)

        roi = frame[self.color_left_top[1]:self.color_right_bot[1],
                    self.color_left_top[0]:self.color_right_bot[0]]
        # Convert the ROI to HSV (OpenCV scale)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Create a mask of pixels that fall within [lower_hsv, upper_hsv]
        mask = cv2.inRange(hsv_roi, lower_hsv_cv, upper_hsv_cv)
        # Calculate the fraction of pixels in the ROI that match our color range
        total_pixels = hsv_roi.shape[0] * roi.shape[1]
        matched_pixels = cv2.countNonZero(mask)
        matched_ratio = matched_pixels / float(total_pixels)

        color_change = False
        # Show live debug info on the color frame
        cv2.putText(frame, f"Ratio: {matched_ratio:.2f} / {self.min_pixels_ratio:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Low: {lower_hsv_cv}  High: {upper_hsv_cv}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if matched_ratio > self.min_pixels_ratio:
            cv2.putText(frame, "Color deviated!", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            color_change = True

        # Record color camera frame and HSV values
        if self._recording:
            if self._color_video_writer is not None:
                self._color_video_writer.write(frame)
            if hasattr(self, '_hsv_data'):
                elapsed_s = time.time() - self._record_start_time
                mean_hsv = cv2.mean(hsv_roi)[:3]
                h_common = float(mean_hsv[0]) * 2.0
                s_common = float(mean_hsv[1]) / 2.55
                v_common = float(mean_hsv[2]) / 2.55
                self._hsv_data.append([elapsed_s, h_common, s_common, v_common])

        return frame, color_change

    def put_label(self, img, label_text):
        """Puts a label at the bottom-right corner of 'img'."""
        if img is None or img.size == 0:
            return img
        
        h, w = img.shape[:2]
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_w, text_h = text_size

        text_x = w - text_w - 10
        text_y = h - 10

        cv2.putText(img, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    def assemble_2x2(self, digit_frame, extracted_img, processed_img, color_frame):
        """Combine frames into a 2×2 layout."""
        images = [digit_frame, extracted_img, processed_img, color_frame]

        max_h, max_w = 0, 0
        for im in images:
            if im is not None and im.size != 0:
                h, w = im.shape[:2]
                max_h = max(max_h, h)
                max_w = max(max_w, w)

        if max_h == 0: max_h = 200
        if max_w == 0: max_w = 200

        def safe_resize(im):
            if im is None or im.size == 0:
                return np.zeros((max_h, max_w, 3), np.uint8)
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            return cv2.resize(im, (max_w, max_h), interpolation=cv2.INTER_AREA)

        digit_frame   = safe_resize(digit_frame)
        extracted_img = safe_resize(extracted_img)
        processed_img = safe_resize(processed_img)
        color_frame   = safe_resize(color_frame)

        digit_frame   = self.put_label(digit_frame, "Digit Frame")
        extracted_img = self.put_label(extracted_img, "Extracted")
        processed_img = self.put_label(processed_img, "Processed")
        color_frame   = self.put_label(color_frame, "Color Frame")

        top_row = np.hstack([digit_frame, extracted_img])
        bottom_row = np.hstack([processed_img, color_frame])
        combined = np.vstack([top_row, bottom_row])

        ch, cw = combined.shape[:2]
        if ch > self.SCREEN_HEIGHT or cw > self.SCREEN_WIDTH:
            ratio = min(self.SCREEN_WIDTH / cw, self.SCREEN_HEIGHT / ch)
            new_w = int(cw * ratio)
            new_h = int(ch * ratio)
            combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return combined

    def start_processing(self):
        """Main loop: read from both cameras, show 2×2 layout."""
        if not self.open_camera():
            print("Failed to open cameras.")
            return

        self.is_running = True
        self.color_change = False
        self.dot_in_center = False

        cv2.namedWindow("MultiCam Display", cv2.WINDOW_NORMAL)

        while self.is_running:
            ret_digit, digit_frame = self.digit_cam.read()
            ret_color, color_frame = self.color_cam.read()

            if not ret_digit or digit_frame is None:
                print("Cannot read from digit camera!")
                break
            if not ret_color or color_frame is None:
                print("Cannot read from color camera!")
                break

            digit_frame, extracted_img, processed_img, self.nn_input_img, self.dot_in_center \
                = self.process_digit_img(digit_frame)
            color_frame, self.color_change = self.process_color_img(color_frame)

            combined_2x2 = self.assemble_2x2(digit_frame, extracted_img, processed_img, color_frame)
            cv2.imshow("MultiCam Display", combined_2x2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

            time.sleep(0.03)

        self.close_camera()

    def start_thread(self):
        if self.thread is not None and self.thread.is_alive():
            print("Camera threads are already running.")
            return
        
        self.thread = threading.Thread(target=self.start_processing, daemon=True)
        self.thread.start()

    def stop_thread(self):
        self.is_running = False
        cv2.waitKey(1)
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        self.thread = None

    def get_data(self):
        """Return the latest detection info."""
        return self.dot_in_center, self.color_change, self.nn_input_img

    def display_UI(self):
        """
        A single function that displays the UI in two rows:
        1) Top row: Start/Stop/Update buttons, Digit ROI, Color ROI
        2) Bottom row: HSV controls, thresholds, etc.
        """
        import ipywidgets as widgets
        from IPython.display import display
        
        # -- Buttons --
        start_button = widgets.Button(description="Start Cameras")
        stop_button  = widgets.Button(description="Stop Cameras")
        update_button = widgets.Button(description="Update Params")

        # -- Digit ROI Widgets --
        _w = widgets.Layout(width='200px')
        digit_left_top_x = widgets.IntText(value=self.digit_left_top[0], description="LeftTop_x", layout=_w)
        digit_left_top_y = widgets.IntText(value=self.digit_left_top[1], description="LeftTop_y", layout=_w)
        digit_right_bot_x = widgets.IntText(value=self.digit_right_bot[0], description="RightBot_x", layout=_w)
        digit_right_bot_y = widgets.IntText(value=self.digit_right_bot[1], description="RightBot_y", layout=_w)

        digit_roi_box = widgets.VBox([
            widgets.HTML("<b>Digit Camera Frame (ROI)</b>"),
            digit_left_top_x,
            digit_left_top_y,
            digit_right_bot_x,
            digit_right_bot_y
        ])

        # -- Color ROI Widgets --
        color_left_top_x = widgets.IntText(value=self.color_left_top[0], description="LeftTop_x", layout=_w)
        color_left_top_y = widgets.IntText(value=self.color_left_top[1], description="LeftTop_y", layout=_w)
        color_right_bot_x = widgets.IntText(value=self.color_right_bot[0], description="RightBot_x", layout=_w)
        color_right_bot_y = widgets.IntText(value=self.color_right_bot[1], description="RightBot_y", layout=_w)

        color_roi_box = widgets.VBox([
            widgets.HTML("<b>Color Camera Frame (ROI)</b>"),
            color_left_top_x, 
            color_left_top_y,
            color_right_bot_x, 
            color_right_bot_y
        ])

        # -- Red-dot & HSV Controls --
        center_x_threshold_widget = widgets.IntText(value=self.center_x_threshold,
                                                    description="Red_X_thres", layout=_w)

        # HSV controls (OpenCV scale or "common" scale depending on your code)
        lower_hue = widgets.IntText(value=self.lower_hsv[0], description="Lower Hue", layout=_w)
        lower_sat = widgets.IntText(value=self.lower_hsv[1], description="Lower Sat", layout=_w)
        lower_val = widgets.IntText(value=self.lower_hsv[2], description="Lower Val", layout=_w)
        upper_hue = widgets.IntText(value=self.upper_hsv[0], description="Upper Hue", layout=_w)
        upper_sat = widgets.IntText(value=self.upper_hsv[1], description="Upper Sat", layout=_w)
        upper_val = widgets.IntText(value=self.upper_hsv[2], description="Upper Val", layout=_w)

        min_pixels_ratio_wid = widgets.FloatText(value=self.min_pixels_ratio,
                                                 description="Min Px Ratio", layout=_w)

        # -- HSV color swatches --
        def _make_swatch_html(h, s, v):
            """Convert common HSV (H:0-360, S:0-100, V:0-100) to an HTML color swatch."""
            import colorsys
            h_norm = max(0.0, min(1.0, h / 360.0))
            s_norm = max(0.0, min(1.0, s / 100.0))
            v_norm = max(0.0, min(1.0, v / 100.0))
            r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)
            r_int, g_int, b_int = int(r * 255), int(g * 255), int(b * 255)
            hex_color = f"#{r_int:02X}{g_int:02X}{b_int:02X}"
            return (
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px;">'
                f'<div style="width:90px;height:36px;background-color:{hex_color};'
                f'border:2px solid #555;border-radius:5px;"></div>'
                f'<span style="font-family:monospace;font-size:12px;line-height:1.5;">'
                f'<b>{hex_color}</b><br>rgb({r_int}, {g_int}, {b_int})</span>'
                f'</div>'
            )

        lower_swatch = widgets.HTML(_make_swatch_html(
            lower_hue.value, lower_sat.value, lower_val.value))
        upper_swatch = widgets.HTML(_make_swatch_html(
            upper_hue.value, upper_sat.value, upper_val.value))

        def _update_lower_swatch(change):
            lower_swatch.value = _make_swatch_html(
                lower_hue.value, lower_sat.value, lower_val.value)

        def _update_upper_swatch(change):
            upper_swatch.value = _make_swatch_html(
                upper_hue.value, upper_sat.value, upper_val.value)

        for w in (lower_hue, lower_sat, lower_val):
            w.observe(_update_lower_swatch, names='value')
        for w in (upper_hue, upper_sat, upper_val):
            w.observe(_update_upper_swatch, names='value')

        threshold_box = widgets.VBox([
            widgets.HTML("<b>Red-dot threshold (pixels):</b>"),
            center_x_threshold_widget,
            widgets.HTML("<b>Color change threshold (0-1):</b>"),
            min_pixels_ratio_wid,
        ])

        hsv_low_box = widgets.VBox([
            widgets.HTML("<b>HSV lower color:</b>"),
            lower_hue, lower_sat, lower_val,
            lower_swatch,
        ])

        hsv_high_box = widgets.VBox([
            widgets.HTML("<b>HSV upper color:</b>"),
            upper_hue, upper_sat, upper_val,
            upper_swatch,
        ])

        # -- Button callbacks --
        def on_start_clicked(b):
            self.start_thread()

        def on_stop_clicked(b):
            self.stop_thread()

        def on_update_clicked(b):
            # Update digit ROI
            self.digit_left_top  = (digit_left_top_x.value, digit_left_top_y.value)
            self.digit_right_bot = (digit_right_bot_x.value, digit_right_bot_y.value)

            # Update color ROI
            self.color_left_top  = (color_left_top_x.value, color_left_top_y.value)
            self.color_right_bot = (color_right_bot_x.value, color_right_bot_y.value)

            # Update red-dot threshold
            self.center_x_threshold = center_x_threshold_widget.value

            # Update HSV range & min_pixels_ratio
            self.lower_hsv = [lower_hue.value, lower_sat.value, lower_val.value]
            self.upper_hsv = [upper_hue.value, upper_sat.value, upper_val.value]
            self.min_pixels_ratio = min_pixels_ratio_wid.value

            print("Parameters updated!")

        # -- Button bindings --
        start_button.on_click(on_start_clicked)
        stop_button.on_click(on_stop_clicked)
        update_button.on_click(on_update_clicked)

        # -- Column 1: Start/Stop/Update Buttons --
        actions_box = widgets.VBox([
            widgets.HTML("<b>Actions</b>"),
            start_button, 
            stop_button,
            update_button
        ])

        # -- Now arrange rows and columns --
        # Top row: (Actions, Digit ROI, Color ROI)
        top_row = widgets.HBox([actions_box, digit_roi_box, color_roi_box])
        # Bottom row: (HSV controls)
        bottom_row = widgets.HBox([threshold_box, hsv_low_box, hsv_high_box])

         # A horizontal rule (divider) to separate top/bottom rows
        hr_line = widgets.HTML("<hr style='border:1px solid #ccc; width:100%'>")

         # -- Final UI: top row, then horizontal rule, then bottom row --
        ui = widgets.VBox([
            top_row,
            hr_line,
            bottom_row,
            hr_line
        ])
        display(ui)


## Black Box Recognition Module

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # Get the four vertices in order
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # Get the transformation matrix and apply perspective transformation
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points, ensure the output is square
    maxDim = max(maxWidth, maxHeight)
    dst = np.array([
        [0, 0],
        [maxDim - 1, 0],
        [maxDim - 1, maxDim - 1],
        [0, maxDim - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxDim, maxDim))

    return warped


# Detect the outline of the black frame and extract it
def detect_and_extract_number_1(image):
    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image: black numbers and frame are white, background is
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations, perform dilation and erosion to correct incomplete edges
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)  # Dilation
    eroded = cv2.erode(dilated, kernel, iterations=1)  # Erosion
    # Find contours after dilation
    contours_dilated, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find contours after erosion
    contours_eroded, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours after dilation
    dilated_image = image.copy()
    cv2.drawContours(dilated_image, contours_dilated, -1, (0, 255, 0), 3)
    # Draw contours after erosion
    eroded_image = image.copy()
    cv2.drawContours(eroded_image, contours_eroded, -1, (0, 0, 255), 3)

    # Find contours and get the hierarchy
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    contour_image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area / image_area > 0.2:
            # Approximate the contour as a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # If the contour is quadrilateral, consider it as a black frame
            if len(approx) == 4:
                # Get the four corner points
                pts = np.array([pt[0] for pt in approx], dtype=np.int32)

                # Draw red dots on the four endpoints
                for pt in pts:
                    cv2.circle(contour_image, tuple(pt), 5, (0, 0, 255), -1)
                # Get the minimum rectangular area of the black frame

                # Perspective transformation to correct to a square
                extracted_image_1 = four_point_transform(image, pts)
                # Return the processed number image
                return extracted_image_1
    return None


# Detect the inner outline black box and extract the digital area inside the black box
def detect_and_extract_number_2(image):
    # Convert to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the image: black numbers and frame are white, background is black
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    # Use Canny edge detection
    edges = cv2.Canny(thresh, 50, 150)

    # Morphological operations, perform dilation and erosion to correct incomplete edges
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)  # Dilation
    eroded = cv2.erode(dilated, kernel, iterations=1)  # Erosion

    # Find contours after dilation
    contours_dilated, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find contours after erosion
    contours_eroded, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours after dilation
    dilated_image = image.copy()
    cv2.drawContours(dilated_image, contours_dilated, -1, (0, 255, 0), 3)
    # Draw contours after erosion
    eroded_image = image.copy()
    cv2.drawContours(eroded_image, contours_eroded, -1, (0, 0, 255), 3)

    # Find contours and get the hierarchy
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    contour_image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area / image_area > 0.2:
            # Approximate the contour as a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # If the contour is quadrilateral, consider it as a black frame
            if len(approx) == 4:
                # Get the four corner points
                pts = np.array([pt[0] for pt in approx], dtype=np.int32)
                # Draw red dots on the four endpoints
                for pt in pts:
                    cv2.circle(contour_image, tuple(pt), 5, (0, 0, 255), -1)
                # Get the minimum rectangular area of the black frame
                x, y, w, h = cv2.boundingRect(pts)
                extracted_image = image[y + 3:y + h - 3, x + 3:x + w - 3]
                # Return the processed number image
                return extracted_image

    return None

def extract_number_img(image):
    extracted_img = None
    extracted_PIL_img = None
    # Feedback whether the black box is detected, and extract the outer outline and inner part of the black box
    extracted_image_1 = detect_and_extract_number_1(image)
    # If a black box is detected, the outer outline and the inner part of the black box will be extracted
    if extracted_image_1 is not None:
        # Feedback whether the black box is detected, and extract the inner outline and inner part of the black box
        extracted_image_2 = detect_and_extract_number_2(extracted_image_1)
        # If a black box is detected, the inner outline and the inner part of the black box will be extracted
        if extracted_image_2 is not None:
            # image process
            # Convert an OpenCV image to RGB mode
            extracted_img = extracted_image_2
            extracted_image_2_rgb = cv2.cvtColor(extracted_image_2, cv2.COLOR_BGR2RGB)
            # Convert an OpenCV image to PIL image
            extracted_PIL_img = Image.fromarray(extracted_image_2_rgb)
    return extracted_img, extracted_PIL_img

### Red Dot Detection Module
def detect_red_dot(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    red_dot = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(red_dot, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            # print("Red dot detected")
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print(f"Red dot coordinates: ({cX}, {cY})")
            return cX, cY, contours
    return None, None, None

def draw_red_dot(frame, red_dot_x, red_dot_y, contours, left_top, right_bot):
    if red_dot_x is not None and red_dot_y is not None:  # Detect the position of the red area
        # Draw the contour of the red area on the image
        # Help debug, show whether the red area is detected
        # cv2.drawContours(frame[y:y + h, x:x + w], contours, -1, (0, 0, 255), 2)
        cv2.drawContours(frame[left_top[1]:right_bot[1], left_top[0]:right_bot[0]],
                         contours, -1, (0, 0, 255), 2)
    return frame

def check_red_dot_center():
    return

def draw_box_extract(ret, frame, left_top, right_bot):
    if not ret:
        print("Cannot open camera!")
        return None

    # Define the coordinates of the border
    # left top corner
    # x, y, w, h = 560, 140, 800, 800  # You can adjust these values as needed
    # Draw the border on the image
    cv2.rectangle(frame, left_top, right_bot, (0, 255, 0), 2)
    # Crop the image within the border
    # Only recognize this part of the image
    roi = frame[left_top[1]:right_bot[1], left_top[0]:right_bot[0]]
    return frame, roi



# Custom color inversion transform
class InvertColor(object):
    def __call__(self, img):
        return 1 - img


# Custom binarization transform
class Binarize(object):
    def __call__(self, img):
        # Convert PIL image to NumPy array
        img_array = np.array(img)
        # Convert image to grayscale
        if len(img_array.shape) == 3:
            img_array = img_array.mean(axis=2)
        # Binarize with threshold 128
        binary_img_array = (img_array > 80).astype(np.uint8) * 255
        # Convert NumPy array back to PIL image
        binary_img = Image.fromarray(binary_img_array)
        return binary_img


def preprocess_to_MNIST(img):
    """
    Dynamically compare width and height, crop the longer side, and finally scale to 28x28.
    :param img: Input image (PIL Image)
    :return: Preprocessed 28x28 image
    """
    if not isinstance(img, Image.Image):
        raise TypeError("Input image must be a PIL image object")
    # Get the width and height of the input image
    width, height = img.size
    # Calculate the size needed for cropping (choose the smaller side as the square crop side length)
    crop_size = min(width, height)
    # Define transform operations: crop and scale
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),  # Crop to square based on the smaller side
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale image (1 channel)
        transforms.Resize((28, 28), interpolation=Image.LANCZOS),  # Scale to 28x28
        Binarize(),  # Binarize
        transforms.ToTensor(),  # Convert to Torch tensor
        InvertColor(),  # Invert color
    ])
    # Apply transformations
    img_tensor = transform(img)  # Shape: (1, 28, 28)

    # Convert tensor to NumPy array for OpenCV display
    processed_img = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)  # Shape: (28, 28), dtype=uint8

    processed_img_resized = cv2.resize(processed_img, (280, 280), interpolation=cv2.INTER_NEAREST)

    # Convert to MNIST-compatible input format (flattened 784x1)
    nn_input_img = img_tensor.view(1, 784).numpy()  # Shape: (1, 784)

    return processed_img, processed_img_resized, nn_input_img

# Camera 
# Function to open the camera
def open_camera(camera_no):
    cap = cv2.VideoCapture(camera_no)  # Camera device number (adjust as needed)
    if not cap.isOpened():
        print("Cannot get video frame!")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot open camera!")
            break
        # Display the real-time frame
        cv2.imshow("Real-time Number Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_control = CameraControl(digit_cam_id=0, color_cam_id=1)
    cam_control.start_processing()  