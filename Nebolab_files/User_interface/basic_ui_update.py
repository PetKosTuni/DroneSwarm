import cv2 as cv  # include opencv library functions in python
import time
import numpy as np
from .nebolab_basic_setup import IMAGE_FRAME_WIDTH, IMAGE_FRAME_HEIGHT, ROBOT_COUNT
from .ui_state_list import StateMap
from .ui_state_list import StateID
from .on_screen_menu import OnScreenMenu
from .object_data import ObjectData
from .camerabased_localization import localize_from_ceiling

import json


class user_interface:

    def __init__(self, window_name='user_interface'):

        self.state_list = StateMap.DICT
        self.stID_current = 0  # Variable to store and check the state of the user-interface
        self.stID_previous = 0

        self.osd = OnScreenMenu()
        self.data = ObjectData()

        # UI for RPW
        
        self.target_point_groups = []  # List of lists to store grouped target points
        self.current_target_points = []  # Temporary list for the currently added points

        self.is_running = True
        self.is_record_clean = False
        self.video_writer = None
        # Create new window and register mouse callback
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        cv.setMouseCallback(window_name, self.mouse_event_callback)
        self.window_name = window_name

        # Mouse state
        self.is_menu_area_clicked = False
        self.mouse_px, self.mouse_py = -1, -1

        # Call the class for localization, for drawing the arrow
        self.loc = localize_from_ceiling(ROBOT_COUNT)

        # UI Coverage Control: Initialize variables for status bar and battery level
        # example battery levels for each robot
        self.battery_levels = [100] * ROBOT_COUNT
        self.hovered_button = None  # To track hovered button
    #RPW
    def finalize_current_group(self):
        if self.current_target_points:
            # Add the current group of points to the list of groups
            self.target_point_groups.append(self.current_target_points)
            print(f"Finalized group: {self.current_target_points}")
            # Clear the current group to start a new one
            self.current_target_points = []
            #debug
            print(f"All groups: {self.target_point_groups}")

            #close the window after saving the points
            cv.destroyWindow("Target Points Input") 
        else:
            print("No points to save.")

    # UI Coverage Control: Render the status bar at the top of the UI
    def render_status_bar(self):
        """Draws a status bar at the top of the UI to show information like battery levels."""
        bar_height = 30
        cv.rectangle(self.img_get, (0, 0),
                     (IMAGE_FRAME_WIDTH, bar_height), (50, 50, 50), -1)
        text = print(f"Current State: {self.stID_current} | Battery Levels: {self.battery_levels}")
        cv.putText(self.img_get, text, (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # UI Coverage Control: Draw battery level indicator
    def render_battery_indicator(self):
        """Draw battery level for each robot as bars or percentages."""
        for i, level in enumerate(self.battery_levels):
            bar_width = int(level * 0.5)
            x = 10
            y = IMAGE_FRAME_HEIGHT - (len(self.battery_levels) - i) * 20 - 30  # Positioning at the bottom
            cv.rectangle(self.img_get, (x, y),
                         (x + bar_width, y + 10), (0, 255, 0), -1)
            cv.putText(self.img_get, f"Robot {i + 1}: {level}%", (x + 60, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # mouse callback function

    def mouse_event_callback(self, event, px, py, flags, param):
        self.mouse_px, self.mouse_py = px, py  # STORE mouse value
        # RESERVE the left-most area and right-most area for menu selection

        # UI for RPW
        if (px < OnScreenMenu.LEFT_BOUND) or (px > OnScreenMenu.RIGHT_BOUND):
            if event == cv.EVENT_LBUTTONDOWN:
                self.is_menu_area_clicked = True
        else:
            self.is_menu_area_clicked = False

        # ADDITION STARTS HERE
        # Only add points if in the correct state (CF_TARGET_POINT_INPUT)
        if event == cv.EVENT_LBUTTONDOWN:  # Left-click event
            current_state_id = self.state_list[self.stID_current].ID
            if current_state_id == StateID.CF_TARGET_POINT_INPUT:
                # Add point to the current group
                self.current_target_points.append((px, py))
                print(f"Point added to current group: ({px}, {py})")
        # ADDITION ENDS HERE

        # Evaluate based on current UI state
        if not self.is_menu_area_clicked:
            self.state_list[self.stID_current].mouse_event(
                event, px, py, self.data)

        # UI Coverage Control: Check for hover effect
        if self.is_hovering_button():
            self.hovered_button = True
        else:
            self.hovered_button = False
    # UI Coverage Control: Helper function to detect hovering over menu items
    def is_hovering_button(self):
        """Returns True if the mouse is hovering over a menu item."""
        for i in range(len(self.state_list)):
            if self.is_menu_clicked('top_left', i):
                return True
        return False

    # supporting function to detect click position
    def is_menu_clicked(self, mode, order):
        if self.is_menu_area_clicked:
            if mode == 'top_left':
                box_left = (OnScreenMenu.FRAME_MIN)
            elif mode == 'top_right':
                box_left = (IMAGE_FRAME_WIDTH -
                            OnScreenMenu.FRAME_MIN - OnScreenMenu.BOX_SIZE_X)
            else:
                return False

            box_right = (box_left + OnScreenMenu.BOX_SIZE_X)
            box_top = (OnScreenMenu.FRAME_MIN +
                       (OnScreenMenu.BOX_SIZE_Y+OnScreenMenu.Y_DIST)*order)
            box_bottom = (box_top + OnScreenMenu.BOX_SIZE_Y)
            return (self.mouse_px >= box_left) and (self.mouse_px <= box_right) \
                and (self.mouse_py >= box_top) and (self.mouse_py <= box_bottom)

        else:
            return False

    # Evaluate whether the keyboard or mouse is used to change menu
    def evaluate_key_and_mouse(self, key):

        #call handle_key_pressed to check if 'f' is pressed

        #self.handle_key_press(key)

        if key == ord('q') or self.is_menu_clicked('top_right', 0):
            cv.destroyAllWindows()
            self.is_running = False  # q to exit

        elif key == ord('t') or self.is_menu_clicked('top_right', 23):
            self.is_record_clean = not self.is_record_clean  # toggle

        elif key == ord('0') or self.is_menu_clicked('top_right', 1):
            self.stID_previous = self.stID_current
            self.stID_current = 0  # 0 to reset to the top menu
            self.state_list[self.stID_current].enter(self.data)

        elif key == ord('z') or self.is_menu_clicked('top_right', 3):
            temp = self.stID_previous
            self.stID_previous = self.stID_current
            self.stID_current = temp  # revert back to previous state
            self.state_list[self.stID_current].enter(self.data)
        
    
        elif key == ord('f') and self.stID_current == StateID.CF_TARGET_POINT_INPUT: # Finalize the current group
            print("Finalizing current group in Target Point Input mode...")
            points = self.state_list[self.stID_current].finalize_current_group()

            # Saving points into JSON file
            with open('points.json', 'w') as out:
                json.dump(points, out)

            self.stID_current = self.stID_previous

        else:  # Inspect based on each current state
            left_osd_list = self.state_list[self.stID_current].OSDLeft
            next = None
            for dict in left_osd_list:
                k, o, a = dict['key'], dict['ord'], dict['act']
                if (next is None) and (k is not None):
                    if key == ord(k) or self.is_menu_clicked('top_left', o):
                        next = a

            # evaluate result
            if next is not None:
                self.stID_previous = self.stID_current
                self.stID_current = next
                self.state_list[self.stID_current].enter(self.data)

        # ALWAYS RESET THE FLAG at the end of checking the mouse click
        self.is_menu_area_clicked = False

    def __check_recording(self, img_get):
        # save this image to be used in recording
        if self.is_record_clean:
            if self.video_writer is None:  # init new location to record
                fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
                v_fname = 'UI_record' + \
                    time.strftime("_%Y%m%d_%H%M%S") + '.avi'
                self.video_writer = cv.VideoWriter(
                    v_fname, fourcc, 30, (IMAGE_FRAME_WIDTH, IMAGE_FRAME_HEIGHT))
            # save frame into video
            self.video_writer.write(img_get)
        else:
            # reset everytime the recording is stopped
            if self.video_writer is not None:
                # Stop & save recording
                self.video_writer.release()
                self.video_writer = None

    def update_user_interface(self, image_frame,
                              poses_center=[None], poses_ahead=[None], scan_range=[None],
                              battery=[None], A_graph=None):
        self.img_get = self.loc.draw_pose_from_data(
            image_frame, poses_ahead, poses_center)

        # UPDATE robot position and visual
        self.data.update_robot_data(poses_center, poses_ahead, scan_range, battery, A_graph)
        self.img_get = self.state_list[self.stID_current].update_visual(
            self.img_get, self.data)
        
        # UI FOR rpw, FIX
        # Draw the targeted points on the image
	    # Draw a small red circle for each target point
        #cv.circle(self.img_get, point, 5, (0, 0, 255), -1)
        #cv.putText(self.img_get, f'({point[0]}, {point[1]})', (point[0] + 10, point[1] - 10),        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

        self.__check_recording(self.img_get)  # clean record without menu

        # UI Coverage Control: Draw status bar and battery indicator
        self.render_status_bar()
        self.render_battery_indicator()

        # Draw on screen display
        self.osd.set_img_canvas(self.img_get)
        self.osd.show_menu_on_screen(
            self.state_list[self.stID_current].OSDLeft)  # draw menu on screen
        if self.is_record_clean:
            # always shown on the right side
            self.osd.draw_top_right_menu(
                23, 't', 'is recording.. click to stop')
        else:
            # always shown on the right side
            self.osd.draw_top_right_menu(23, 't', 'start recording')

        # Show the visual
        cv.imshow(self.window_name, self.img_get)  # show captured frame

        # Evaluate user input
        key = cv.waitKey(1) & 0XFF
        self.evaluate_key_and_mouse(key)

    def generate_ui_numpy_data(self):
        # construct the obstacle in the form of x, y, and rad
        n = len(self.data.list_obst)
        np_obst = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            np_obst[i, 0] = self.data.list_obst[i].x
            np_obst[i, 1] = self.data.list_obst[i].y
            np_obst[i, 2] = self.data.list_obst[i].rad

        # construct the goals in the form of x, y
        n = len(self.data.list_goal)
        np_goal = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            np_goal[i, 0] = self.data.list_goal[i].x
            np_goal[i, 1] = self.data.list_goal[i].y

        np_adj_attack_mat = self.data.adj_attack_mat.copy()

        return np_obst.flatten(), np_goal.flatten(), \
            np_adj_attack_mat.flatten()


if __name__ == '__main__':
    # vidcap = cv.VideoCapture('video_test_robot.avi')
    ret, img = vidcap.read()  # capture a frame from live video

    ui = user_interface('Visualization with user_interface')

    from .camerabased_localization import localize_from_ceiling
    # Call the class for localization
    loc = localize_from_ceiling()

    if ret:
        while (ui.is_running):
            poses_center, poses_ahead = loc.localize_all_robots(img)
            img = loc.draw_pose(img)

            # 360 data for 4 robots with all 0 values
            dummy_LIDAR_data = [[0]*360]*4
            ui.update_user_interface(
                img, poses_center, poses_ahead, dummy_LIDAR_data)
            
            # Generate numpy data to publish
            np_obst, np_goal = ui.generate_ui_numpy_data()
