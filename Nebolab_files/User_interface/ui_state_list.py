import cv2 as cv #include opencv library functions in python
from .nebolab_basic_setup import pos_m2pxl, SCALE_M2PXL
from .object_data import Circle
import numpy as np

import subprocess

# -----------------------------------------------------------------
# Below we define the ID to asses the state of the UI
# The depth-level of the menu is indicated by the number of digit
# NOTE: each of the ID has to be a unique integer


class StateID:
    CLEAN_STATE = 0
    SHOW_MENU = 1 # m key-press

    MODIFY_SCENARIO = 10 # s key-press
    SCENARIO_4FORM_OBST = 11 # 1 key-press
    SCENARIO_2FORM_AVOID = 12 # 2 key-press
    SCENARIO_RESILIENT = 13 # 3 key-press
    SCENARIO_RPW_2 = 14 # 4 key-press
    SCENARIO_RPW_4 = 15 # 5 key-press
    SCENARIO_DISTC_1 = 16 # 6 key-press
    SCENARIO_FLEXFORM = 17 # 7 key-press
    SCENARIO_CF = 18 # 8 key-press
    SCENARIO_RESILIENT_UI = 19 # 9 key-press
    SCENARIO_ERGODIC = 20 # 10 key press

    MOD_DRAW_GOAL = 111 # g key-press
    MOD_DRAW_OBSTACLE = 112 # d key-press
    MOD_CHANGE_OBSTACLE_RADIUS = 113 # r key-press
    MOD_MOVE_OBJECT = 114 # m key-press
    MOD_DELETE_OBJECT = 115 # x key-press
    MOD_COMMUNICATION = 116 # c key-press
    MOD_MANUAL_CONTROL = 117 # o key-press

    CF_TAKEOFF_ALL = 118 # , key press
    CF_LANDING_ALL = 119 # . key press

    CF_ERGO_TAKEOFF = 120 # ( key press
    CF_ERGO_LANDING = 121 # ) key press

    
    CF_SAVE_ALL = 122 ## / key press  <- New action
    CF_TARGET_POINT_INPUT = 123 #new state for targeted points
   

class DefState:
    def __init__(self): 
        self.ID, self.OSDLeft, self.OSDRight = 0, [], []
    def enter(self, object_list): pass # Reinitialization when entering this state
    def mouse_event(self, event, px, py, object_list): pass # Processing mouse events
    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        return img_main # specific drawing on each state

# ----------------------------------------------------------------------------------------------------

class StCleanState(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.CLEAN_STATE
        self.OSDLeft = [ 
            { 'ord':0, 'key':'m', 'msg':'enter menu', 'act':StateID.SHOW_MENU } ]

class StShowMenu(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SHOW_MENU
        self.OSDLeft = [
            { 'ord':0, 'key':'s', 'msg':'modify scenario',          'act':StateID.MODIFY_SCENARIO },
            { 'ord':1, 'key':'1', 'msg':'4 robots formation',       'act':StateID.SCENARIO_4FORM_OBST },
            { 'ord':2, 'key':'2', 'msg':'2 formation avoidance',    'act':StateID.SCENARIO_2FORM_AVOID },
            { 'ord':3, 'key':'3', 'msg':'Resilient scenario',       'act':StateID.SCENARIO_RESILIENT },
            { 'ord':4, 'key':'4', 'msg':'RPW with 2 robots',        'act':StateID.SCENARIO_RPW_2 },
            { 'ord':5, 'key':'5', 'msg':'RPW with 4 robots',        'act':StateID.SCENARIO_RPW_4 },
            { 'ord':6, 'key':'6', 'msg':'Dist Control 1 robot',     'act':StateID.SCENARIO_DISTC_1 },
            { 'ord':7, 'key':'7', 'msg':'Rigid-Elastic Formation',  'act':StateID.SCENARIO_FLEXFORM },
            { 'ord':8, 'key':'8', 'msg':'Crazyswarm with 6 drones', 'act':StateID.SCENARIO_CF },
            { 'ord':9, 'key':'9', 'msg':'Resilient LF with UI',     'act':StateID.SCENARIO_RESILIENT_UI },
            { 'ord':10, 'key':'a', 'msg':'Ergodic control',         'act':StateID.SCENARIO_ERGODIC },
            
        ]

# ----------------------------------------------------------------------------------------------------

class StModifyScenario(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MODIFY_SCENARIO
        self.OSDLeft = [
            { 'ord':0, 'key':'g', 'msg':'draw goal',                  'act':StateID.MOD_DRAW_GOAL },
            { 'ord':1, 'key':'d', 'msg':'draw obstacle',              'act':StateID.MOD_DRAW_OBSTACLE },
            { 'ord':2, 'key':'r', 'msg':'change obstacle radius',     'act':StateID.MOD_CHANGE_OBSTACLE_RADIUS },
            { 'ord':3, 'key':'m', 'msg':'move object',                'act':StateID.MOD_MOVE_OBJECT },
            { 'ord':4, 'key':'x', 'msg':'delete object',              'act':StateID.MOD_DELETE_OBJECT },
            { 'ord':5, 'key':'c', 'msg':'modify communication',       'act':StateID.MOD_COMMUNICATION },
            { 'ord':6, 'key':'o', 'msg':'manualy control each robot', 'act':StateID.MOD_MANUAL_CONTROL },
        ]


class StModDrawGoal(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_DRAW_GOAL
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to draw goal', 'act':None } ]
        # save mouse info
        self.m_px, self.m_py = 0, 0

    def mouse_event(self, mouse_event, px, py, object_list): 
        self.m_px, self.m_py = px, py
        # One click to place the goal
        if mouse_event == cv.EVENT_LBUTTONDOWN: object_list.add_goal(px, py)

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        return img_main # specific drawing on each state


class StModDrawObstacle(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_DRAW_OBSTACLE
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to draw obstacle', 'act':None } ]
        # Additional variable
        self.__created_obst_id = None
        self.m_px, self.m_py = 0, 0 # save mouse info

    def mouse_event(self, mouse_event, px, py, object_list): 
        self.m_px, self.m_py = px, py
        # First click to select the place to draw
        # Move mouse cursor to adjust the obstacle size
        # Second click to confirm and finalize
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            if (self.__created_obst_id is None): # assign obstacle with default value on first click
                self.__created_obst_id = object_list.add_obstacle(px, py)
            else: self.__created_obst_id = None # second click, radius is already selected
        elif (mouse_event == cv.EVENT_MOUSEMOVE) and (self.__created_obst_id is not None): # it is assigned some value
            object_list.change_obst_i_rad_to(self.__created_obst_id, px, py)

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        if (self.__created_obst_id is None): 
            object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        else: 
            object_list.show_mouse_dist_from_obst(img_main, self.__created_obst_id, self.m_px, self.m_py)
        return img_main # specific drawing on each state


class StModChangeObstacleRadius(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_CHANGE_OBSTACLE_RADIUS
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to change obstacle radius', 'act':None } ]
        # Additional variable
        self.__selected_obst_id = None
        self.m_px, self.m_py = 0, 0 # save mouse info

    def mouse_event(self, mouse_event, px, py, object_list): 
        self.m_px, self.m_py = px, py
        # First click to select the obstacle to adjust
        # Move mouse cursor to adjust the obstacle size
        # Second click to confirm and finalize
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            if (self.__selected_obst_id is None): # assign obstacle with default value on first click
                id = object_list.get_id_selected_obst(px, py)
                if id is not None: self.__selected_obst_id = id
            else: self.__selected_obst_id = None # second click, radius is already selected
            
        elif (mouse_event == cv.EVENT_MOUSEMOVE) and (self.__selected_obst_id is not None): # it is assigned some value
            object_list.change_obst_i_rad_to(self.__selected_obst_id, px, py)


    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        if (self.__selected_obst_id is not None): 
            object_list.show_mouse_dist_from_obst(img_main, self.__selected_obst_id, self.m_px, self.m_py)
        return img_main # specific drawing on each state


class StModMoveObject(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_MOVE_OBJECT
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'drag and drop to move the object', 'act':None } ]
        # Additional variable
        self.__sel_type = None
        self.__sel_id = None
        self.m_px, self.m_py = 0, 0 # save mouse info

    def mouse_event(self, mouse_event, px, py, object_list): 
        self.m_px, self.m_py = px, py
        # Click and hold left click to move the object (goal / obstacle)
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            self.__sel_type, self.__sel_id = object_list.get_type_id_click(px,py)
        elif (mouse_event == cv.EVENT_MOUSEMOVE) and (self.__sel_id is not None): # it is assigned some value
            object_list.move_type_id(self.__sel_type, self.__sel_id, px, py)
        elif mouse_event == cv.EVENT_LBUTTONUP:
            self.__sel_type, self.__sel_id = None, None # reset the value

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        return img_main # specific drawing on each state


class StModDeleteObject(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_DELETE_OBJECT
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to delete the object', 'act':None } ]

    def mouse_event(self, mouse_event, px, py, object_list): 
        # One click to delete the object (obstacle and goals)
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            object_list.delete_any_obj(px, py)

# TODO: continue from here
class StModCommunication(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_COMMUNICATION
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to select first robot and second click for second robot', 
                'act':None } ]
        # Additional variable
        self.__selected_first_id = None

    def mouse_event(self, mouse_event, px, py, object_list): 
        # First click to select a robot
        # Second click on another robot to finalize the communication to toggle (on or off)
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            id = object_list.get_id_selected_robot(px, py)
            if id is not None:
                if (self.__selected_first_id is None): self.__selected_first_id = id
                else:
                    object_list.toggle_adjacency_matrix(self.__selected_first_id, id)
                    self.__selected_first_id = None

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        if self.__selected_first_id is not None:
            object_list.highlight_robot(img_main, self.__selected_first_id)
        return img_main # specific drawing on each state


class StModManualControl(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.MOD_MANUAL_CONTROL
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to select/unselect the robot and move the cursor to send input', 'act':None } ]
        # Additional variable
        self.__selected_robot_id = None
        self.m_px, self.m_py = 0, 0 # save mouse info

    def mouse_event(self, mouse_event, px, py, object_list): 
        self.m_px, self.m_py = px, py
        # First click to select the robot to move
        # move the cursor to direct the moving direction
        # click the robot again to de-select the robot
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            id = object_list.get_id_selected_robot(px, py)
            if id is not None:
                if (self.__selected_robot_id is None): self.__selected_robot_id = id
                else: self.__selected_robot_id = None

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        if (self.__selected_robot_id is not None): 
            object_list.compute_human_input(img_main, self.__selected_robot_id, self.m_px, self.m_py)
        return img_main # specific drawing on each state


# ----------------------------------------------------------------------------------------------------

class StScen4FormObst(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_4FORM_OBST
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'CCTA-2022 - Four-robot Formation with Obstacle Avoidance', 
                'act':None } ]

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add obstacle
        px_ob1, py_ob1 = pos_m2pxl(-0.4, 0.6)
        px_ob2, py_ob2 = pos_m2pxl(-0.5, -1.0)
        prad_ob = int(0.2*SCALE_M2PXL)
        object_list.add_obstacle( px_ob1, py_ob1, prad_ob)
        object_list.add_obstacle( px_ob2, py_ob2, prad_ob)
        # Add goals
        px_g, py_g = pos_m2pxl(1.4, -0.2)
        object_list.add_goal( px_g, py_g ) 
        # Adjust communication
        A = np.array([ [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0] ])
        object_list.set_adjacency_matrix(A)


class StScen2FormObst(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_2FORM_AVOID
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'CCTA-2022 - Two-robot Formation with Collision Avoidance', 
                'act':None } ]

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add goals
        px_g1, py_g1 = pos_m2pxl(1., -1.)
        px_g2, py_g2 = pos_m2pxl(1., 1.)
        object_list.add_goal( px_g1, py_g1 ) 
        object_list.add_goal( px_g2, py_g2 ) 
        # Adjust communication
        A = np.array([ [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0] ])
        object_list.set_adjacency_matrix(A)

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_object(img_main)
        img_main = object_list.draw_ellipse_formation_2robot(img_main, 0, 1)
        img_main = object_list.draw_ellipse_formation_2robot(img_main, 2, 3)
        return img_main # specific drawing on each state



class StResilient(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_RESILIENT
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'left-click to draw goal', 'act':None } ]

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add goals
        px_g1, py_g1 = pos_m2pxl(0.5, 0.)
        px_g2, py_g2 = pos_m2pxl(0.5, -0.8)
        px_g3, py_g3 = pos_m2pxl(0.5, 0.8)
        object_list.add_goal( px_g1, py_g1 ) 
        object_list.add_goal( px_g2, py_g2 ) 
        object_list.add_goal( px_g3, py_g3 ) 
        # Adjust communication
        # A = np.array([ [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0] ])
        # object_list.set_adjacency_matrix(A)

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, multicolor = True)
        return img_main # specific drawing on each state



class StRPW_2(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_RPW_2
        self.OSDLeft = [  { 'ord':0, 'key':None, 'msg':'RPW with 2 robots', 'act':None } ]
        # formation shift from center in polar coordinate 
        self.form_rad = 0.2
        self.form_rot_shift = 0.
        self.form_theta = np.array([np.pi/2, -np.pi/2])

        c_px, c_py = pos_m2pxl(0, 0.)
        a_px, a_py = pos_m2pxl(self.form_rad, 0.)
        self.form_centroid = Circle( c_px, c_py, 15 )
        self.arrow_head = Circle( a_px, a_py, 15 )

        self.m_px, self.m_py = 0, 0 # save mouse info
        self.is_arrow_tail_selected = False
        self.is_arrow_head_selected = False

    def _get_goal_i(self, i):
        x = self.form_centroid.x + self.form_rad * np.cos(self.form_theta[i] + self.form_rot_shift)
        y = self.form_centroid.y + self.form_rad * np.sin(self.form_theta[i] + self.form_rot_shift)
        return x, y

    def _update_all_goal(self, object_list):
        # Update the rotating info
        arrow_dx  = self.arrow_head.x - self.form_centroid.x
        arrow_dy  = self.arrow_head.y - self.form_centroid.y
        self.form_rot_shift = np.arctan2(arrow_dy, arrow_dx)
        # Update the scaling info
        self.form_rad = np.sqrt( arrow_dx**2 + arrow_dy**2 )
        # update the remaining info
        for i in range( len(self.form_theta) ):
            x, y = self._get_goal_i(i)
            px_g, py_g = pos_m2pxl(x, y)
            object_list.move_goal_id(i, px_g, py_g)

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add goals
        for _ in range( len(self.form_theta) ): object_list.add_goal( 1, 1 ) 
        self._update_all_goal(object_list)

    def mouse_event(self, mouse_event, px, py, object_list): 
        # Drag and drop Arrow's head and tail to adjust the formation
        self.m_px, self.m_py = px, py
        # Click and hold left click to move the object (goal / obstacle)
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            self.is_arrow_tail_selected = self.form_centroid.is_contain_pxl(px, py)
            self.is_arrow_head_selected = self.arrow_head.is_contain_pxl(px, py)

        elif (mouse_event == cv.EVENT_MOUSEMOVE): # it is assigned some value
            if self.is_arrow_tail_selected:
                move_x = px - self.form_centroid.px
                move_y = py - self.form_centroid.py
                self.form_centroid.update_center_pos(px, py)
                # update arrow head as well
                new_px = self.arrow_head.px + move_x
                new_py = self.arrow_head.py + move_y
                self.arrow_head.update_center_pos(new_px, new_py)
                # update all goals
                self._update_all_goal(object_list)

            elif self.is_arrow_head_selected: 
                self.arrow_head.update_center_pos(px, py)
                self._update_all_goal(object_list)
        
        elif mouse_event == cv.EVENT_LBUTTONUP: # reset the value
            self.is_arrow_tail_selected = False
            self.is_arrow_head_selected = False
        

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, multicolor = True)
        # object_list.draw_scan_range(img_main)
        # Draw circle to edit the parameter
        cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), self.form_centroid.prad, (255,255,255), 2)
        cv.circle(img_main, (self.arrow_head.px, self.arrow_head.py), self.arrow_head.prad, (255,255,255), 2)
        # Draw the value during editing
        if self.is_arrow_tail_selected: 
            object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        if self.is_arrow_head_selected:
            cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), 
                int(self.form_rad*SCALE_M2PXL), (255,255,255), 1)
            # show len and angle
            mouse_text = "r: {:.2f}m, {:.2f}deg".format(self.form_rad, np.rad2deg(self.form_rot_shift))
            cv.putText(img_main, mouse_text, (self.m_px, self.m_py), 
                object_list.text_font, 1, object_list.text_color, 2, cv.LINE_AA) # show the key press

        return img_main # specific drawing on each state


class StRPW_4(StRPW_2):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_RPW_4
        self.OSDLeft = [  { 'ord':0, 'key':None, 'msg':'RPW with 4 robots', 'act':None } ]
        # formation shift from center
        # formation shift from center in polar coordinate 
        self.form_rad = 0.4
        self.form_theta = np.array([0, np.pi/2, np.pi, -np.pi/2 ]) + np.pi/4


class StDist_1(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_RPW_2
        self.OSDLeft = [  { 'ord':0, 'key':None, 'msg':'DistCourse with 1 robot', 'act':None } ]

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add goals
        px_g, py_g = pos_m2pxl(1.5, 1.)
        object_list.add_goal( px_g, py_g ) 

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, multicolor = True)
        object_list.draw_batt_status(img_main)
        # Draw custom circle
        px, py = pos_m2pxl(0., 0.1)
        outer_prad = int((0.7+0.2)*SCALE_M2PXL)
        prad = int(0.7*SCALE_M2PXL)
        cv.circle(img_main, (px,py), outer_prad, (0,255,0) ,1) 
        cv.circle(img_main, (px,py), prad , (0,0,255) ,-1) 

        return img_main # specific drawing on each state


class StScenFlexForm(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_FLEXFORM
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'RAL-2024 - Four-robot Formation with Rigid and Elastic Edges', 
                'act':None } ]

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Adjust communication
        # A = np.array([ [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0] ])
        A = np.array([ [0, 10, 1, 1], [10, 0, 1, 0], [1, 1, 0, 10], [1, 0, 10, 0] ])
        object_list.set_adjacency_matrix(A)

class CF(StRPW_2):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_CF
        self.OSDLeft = [  
            { 'ord':0, 'key':None, 'msg':'Crazyswarm with 6 drones', 'act':None },
            { 'ord':1, 'key':',', 'msg':'Take off all', 'act':StateID.CF_TAKEOFF_ALL },
            { 'ord':2, 'key':'.', 'msg':'Land all', 'act':StateID.CF_LANDING_ALL },
            {'ord': 3, 'key': '/', 'msg': 'Hover All', 'act': StateID.CF_SAVE_ALL}  # New entry
        ]
        # formation shift from center
        # formation shift from center in polar coordinate 
        self.form_rad = 1.0
        self.form_rot_shift = 0.
        self.form_theta = np.array([2*np.pi/3, np.pi/3, 0, np.pi, -2*np.pi/3, -np.pi/3]) + np.pi/6
        c_px, c_py = pos_m2pxl(0, 0.)
        a_px, a_py = pos_m2pxl(self.form_rad, 0.)
        self.form_centroid = Circle( c_px, c_py, 15 )
        self.arrow_head = Circle( a_px, a_py, 15 )
        


    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, cfcolor = True)
        # object_list.draw_scan_range(img_main)
        # Draw circle to edit the parameter
        cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), self.form_centroid.prad, (255,255,255), 2)
        cv.circle(img_main, (self.arrow_head.px, self.arrow_head.py), self.arrow_head.prad, (255,255,255), 2)
        # Draw the value during editing
        if self.is_arrow_tail_selected: 
            object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        if self.is_arrow_head_selected:
            cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), 
                int(self.form_rad*SCALE_M2PXL), (255,255,255), 1)
            # show len and angle
            mouse_text = "r: {:.2f}m, {:.2f}deg".format(self.form_rad, np.rad2deg(self.form_rot_shift))
            cv.putText(img_main, mouse_text, (self.m_px, self.m_py), 
                object_list.text_font, 1, object_list.text_color, 2, cv.LINE_AA) # show the key press

        return img_main # specific drawing on each state

class StTakeOff(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.CF_TAKEOFF_ALL
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'Taking off all drones', 'act':None } ]
        
    def enter(self, object_list):
        takeoff_call = subprocess.Popen([
                                            "ros2",
                                            "service",
                                            "call",
                                            "/nebolab/allcfs/takeoff", 
                                            "std_srvs/srv/Trigger", 
                                            "{}"
                                        ],
                                        stdout=subprocess.PIPE,
                                        close_fds=True)
        takeoff_call.communicate()[0]

class StLanding(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.CF_LANDING_ALL
        self.OSDLeft = [ 
            { 'ord':0, 'key':None, 'msg':'landing all drones', 'act':None } ]
        
        
    def enter(self, object_list):
        landing_call = subprocess.Popen([
                                            "ros2",
                                            "service",
                                            "call",
                                            "/nebolab/allcfs/land", 
                                            "std_srvs/srv/Trigger", 
                                            "{}"
                                        ],
                                        stdout=subprocess.PIPE,
                                        close_fds=True)
        landing_call.communicate()[0]


class StResilientUI(DefState):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_RPW_2
        self.OSDLeft = [  { 'ord':0, 'key':None, 'msg':'Resilient LF with UI', 'act':None } ]
        # formation shift from center in polar coordinate 
        self.form_rad = 0.4
        self.form_rot_shift = 0.
        self.form_theta = np.array([0, np.pi/2, np.pi, -np.pi/2 ]) + np.pi/4

        c_px, c_py = pos_m2pxl(0, 0.)
        a_px, a_py = pos_m2pxl(self.form_rad, 0.)
        self.form_centroid = Circle( c_px, c_py, 15 )
        self.arrow_head = Circle( a_px, a_py, 15 )

        self.m_px, self.m_py = 0, 0 # save mouse info
        self.is_arrow_tail_selected = False
        self.is_arrow_head_selected = False

    def _get_goal_i(self, i):
        x = self.form_centroid.x + self.form_rad * np.cos(self.form_theta[i] + self.form_rot_shift)
        y = self.form_centroid.y + self.form_rad * np.sin(self.form_theta[i] + self.form_rot_shift)
        return x, y

    def _update_all_goal(self, object_list):
        # Update the rotating info
        arrow_dx  = self.arrow_head.x - self.form_centroid.x
        arrow_dy  = self.arrow_head.y - self.form_centroid.y
        self.form_rot_shift = np.arctan2(arrow_dy, arrow_dx)
        # Update the scaling info
        self.form_rad = np.sqrt( arrow_dx**2 + arrow_dy**2 )
        # update the remaining info
        for i in range( len(self.form_theta) ):
            x, y = self._get_goal_i(i)
            px_g, py_g = pos_m2pxl(x, y)
            object_list.move_goal_id(i, px_g, py_g)

    def enter(self, object_list): # Reinitialization when entering this state
        object_list.clear_all_obst()
        object_list.clear_all_goal()
        # Add goals
        for _ in range( len(self.form_theta) ): object_list.add_goal( 1, 1 ) 
        self._update_all_goal(object_list)

    def mouse_event(self, mouse_event, px, py, object_list): 
        # Drag and drop Arrow's head and tail to adjust the formation
        self.m_px, self.m_py = px, py
        # Click and hold left click to move the object (goal / obstacle)
        if mouse_event == cv.EVENT_LBUTTONDOWN:
            self.is_arrow_tail_selected = self.form_centroid.is_contain_pxl(px, py)
            self.is_arrow_head_selected = self.arrow_head.is_contain_pxl(px, py)
            # Detect selection on communication link
            object_list.detect_attack_resilient(px, py)

        elif (mouse_event == cv.EVENT_MOUSEMOVE): # it is assigned some value
            if self.is_arrow_tail_selected:
                move_x = px - self.form_centroid.px
                move_y = py - self.form_centroid.py
                self.form_centroid.update_center_pos(px, py)
                # update arrow head as well
                new_px = self.arrow_head.px + move_x
                new_py = self.arrow_head.py + move_y
                self.arrow_head.update_center_pos(new_px, new_py)
                # update all goals
                self._update_all_goal(object_list)

            elif self.is_arrow_head_selected: 
                self.arrow_head.update_center_pos(px, py)
                self._update_all_goal(object_list)
        
        elif mouse_event == cv.EVENT_LBUTTONUP: # reset the value
            self.is_arrow_tail_selected = False
            self.is_arrow_head_selected = False
        

    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, multicolor = True)
        # object_list.draw_scan_range(img_main)
        # Draw circle to edit the parameter
        cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), self.form_centroid.prad, (255,255,255), 2)
        cv.circle(img_main, (self.arrow_head.px, self.arrow_head.py), self.arrow_head.prad, (255,255,255), 2)
        # Draw the value during editing
        if self.is_arrow_tail_selected: 
            object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        if self.is_arrow_head_selected:
            cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), 
                int(self.form_rad*SCALE_M2PXL), (255,255,255), 1)
            # show len and angle
            mouse_text = "r: {:.2f}m, {:.2f}deg".format(self.form_rad, np.rad2deg(self.form_rot_shift))
            cv.putText(img_main, mouse_text, (self.m_px, self.m_py), 
                object_list.text_font, 1, object_list.text_color, 2, cv.LINE_AA) # show the key press

        object_list.draw_connectivity_resilient(img_main)

        return img_main # specific drawing on each state

class RPW(StRPW_2):
    def __init__(self): 
        super().__init__() 
        self.ID = StateID.SCENARIO_ERGODIC
        self.OSDLeft = [  
            { 'ord':0, 'key':None, 'msg':'Crazyswarm', 'act':None },
            {'ord': 1, 'key': 'p', 'msg': 'Enter Targeted Points Mode', 'act': StateID.CF_TARGET_POINT_INPUT },
            ]
        


    def update_visual(self, img_main, object_list): 
        object_list.draw_all_goal(img_main, cfcolor = True)
        # object_list.draw_scan_range(img_main)
        # Draw circle to edit the parameter
        cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), self.form_centroid.prad, (255,255,255), 2)
        cv.circle(img_main, (self.arrow_head.px, self.arrow_head.py), self.arrow_head.prad, (255,255,255), 2)
        # Draw the value during editing
        if self.is_arrow_tail_selected: 
            object_list.show_mouse_pos(img_main, self.m_px, self.m_py)
        if self.is_arrow_head_selected:
            cv.circle(img_main, (self.form_centroid.px, self.form_centroid.py), 
                int(self.form_rad*SCALE_M2PXL), (255,255,255), 1)
            # show len and angle
            mouse_text = "r: {:.2f}m, {:.2f}deg".format(self.form_rad, np.rad2deg(self.form_rot_shift))
            cv.putText(img_main, mouse_text, (self.m_px, self.m_py), 
                object_list.text_font, 1, object_list.text_color, 2, cv.LINE_AA) # show the key press

        return img_main # specific drawing on each state



# ----------------------------------------------------------------------------------------------------

#UI RPW
#UI RPW




class StTargetPointInput(RPW):
    def __init__(self):
        super().__init__()
        self.ID = StateID.CF_TARGET_POINT_INPUT
        # self.OSDLeft = [
        #     {'ord': 0, 'key': None, 'msg': 'Left-click to add target points', 'act': None}
        #     ]

        self.img_target_input = None  # Initialize the target input image here

        self.current_target_points = []  # Initialize the attribute to store points
        self.target_point_groups = []  # List to store groups of points

    def enter(self, object_list):
        # Create a new window for target point input
        self.current_target_points = []
        cv.namedWindow("Target Points Input", cv.WINDOW_NORMAL)
        cv.resizeWindow("Target Points Input", 800, 600)  # Adjust size as necessary
        self.img_target_input = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background

        # Draw grids on the background including origin
        self._draw_grid(self.img_target_input)

        # Set mouse callback to capture clicks inside the target window
        cv.setMouseCallback("Target Points Input", self.mouse_event, param=None)

        # Show the initial image with the grid
        cv.imshow("Target Points Input", self.img_target_input)

    def mouse_event(self, event, px, py, flags, param):
        # Add points to the list when left mouse button is clicked
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"State: CF_TARGET_POINT_INPUT, Adding Target Point at ({px}, {py})")
            if self.img_target_input is not None:

                px_loc = round((px - 400) / 166, 1)
                py_loc = round((-py + 300) / 125, 1)

                # Update the image with the new target point
                cv.circle(self.img_target_input, (px, py), 5, (0, 0, 255), -1)
                
                # Optionally add coordinates as text
                cv.putText(self.img_target_input, f'({px_loc}, {py_loc})', (px + 10, py - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

                # Refresh the target input window with updated points
                cv.imshow("Target Points Input", self.img_target_input)
                cv.waitKey(1)  # Ensure the window refreshes to show the changes

                # Store the coordinates in the list
                self.current_target_points.append([px_loc + 1.2, py_loc + 1.2])

    #test
    def finalize_current_group(self):
        """ Finalize and save the current group of points """
        if self.current_target_points:
            # Save the current group of points to the list of groups
            self.target_point_groups.append(self.current_target_points)
            print(f"Saved points group: {self.current_target_points}")
            # Clear the current points to prepare for the next group
            self.current_target_points = []
            print(f"All saved points groups: {self.target_point_groups}")
            
            # Close the window after saving the points
            cv.destroyWindow("Target Points Input")
            return self.target_point_groups
        else:
            print("No points to save.")

    #test
    def evaluate_key_and_mouse(self, key):
    # Only process 'f' when in the Targeted Points Mode
        if key == ord('f') and self.stID_current == StateID.CF_TARGET_POINT_INPUT:
            print("Finalizing current group in Target Point Input mode...")
            self.finalize_current_group()
    

    def update_visual(self, img_main, object_list):
        # Keep the main visualization window unchanged
        return img_main

    def _draw_grid(self, image):
        # Find the origin in the center of the window
        origin_x = image.shape[1] // 2
        origin_y = image.shape[0] // 2

        # Draw vertical and horizontal grid lines every 25 pixels
        for x in range(0, image.shape[1], 200):
            cv.line(image, (x, 0), (x, image.shape[0]), (200, 200, 200), 1)
        for y in range(0, image.shape[0], 150):
            cv.line(image, (0, y), (image.shape[1], y), (200, 200, 200), 1)

        # Draw thicker lines for the x and y axes to represent the origin
        cv.line(image, (origin_x, 0), (origin_x, image.shape[0]), (0, 0, 0), 2)  # Y-axis
        cv.line(image, (0, origin_y), (image.shape[1], origin_y), (0, 0, 0), 2)  # X-axis

        # Draw a circle to mark the origin clearly
        cv.circle(image, (origin_x, origin_y), 7, (0, 0, 255), -1)
        # Optionally, add a label for the origin point
        cv.putText(image, "(0, 0)", (origin_x + 10, origin_y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)




class StateMap:
    DICT = {
        StateID.CLEAN_STATE : StCleanState(),
        StateID.SHOW_MENU : StShowMenu(),

        StateID.MODIFY_SCENARIO : StModifyScenario(),
        StateID.SCENARIO_4FORM_OBST : StScen4FormObst(),
        StateID.SCENARIO_2FORM_AVOID : StScen2FormObst(),
        StateID.SCENARIO_RESILIENT : StResilient(),
        StateID.SCENARIO_RPW_2 : StRPW_2(),
        StateID.SCENARIO_RPW_4 : StRPW_4(),
        StateID.SCENARIO_DISTC_1 : StDist_1(),
        StateID.SCENARIO_FLEXFORM : StScenFlexForm(),
        StateID.SCENARIO_CF : CF(),
        StateID.SCENARIO_RESILIENT_UI : StResilientUI(),
        StateID.SCENARIO_ERGODIC: RPW(),
        
        StateID.MOD_DRAW_GOAL : StModDrawGoal(),
        StateID.MOD_DRAW_OBSTACLE : StModDrawObstacle(),
        StateID.MOD_CHANGE_OBSTACLE_RADIUS : StModChangeObstacleRadius(),
        StateID.MOD_MOVE_OBJECT : StModMoveObject(),
        StateID.MOD_DELETE_OBJECT : StModDeleteObject(),
        StateID.MOD_COMMUNICATION : StModCommunication(),
        StateID.MOD_MANUAL_CONTROL : StModManualControl(),

        StateID.CF_TAKEOFF_ALL : StTakeOff(),
        StateID.CF_LANDING_ALL : StLanding(),
        
        StateID.CF_TARGET_POINT_INPUT: StTargetPointInput(),  # Added for UI RPW
    }
    