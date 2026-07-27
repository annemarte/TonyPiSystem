#!/usr/bin/python3
# coding=utf8
# 4.Advanced Lessons\11.Athletics Sport Lesson\Lesson2 Go Up and Down Stair
import os
import sys
import cv2
import time
import math
import threading
import numpy as np
import hiwonder.ros_robot_controller_sdk as rrc
from hiwonder.Controller import Controller
import hiwonder.PID as PID
import hiwonder.Misc as Misc
import hiwonder.Camera as Camera
import hiwonder.ActionGroupControl as AGC
import hiwonder.yaml_handle as yaml_handle

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

# go up and down stair

go_forward = 'go_forward'
go_forward_one_step = 'go_forward_one_step'
go_forward_one_small_step = 'go_forward_one_small_step'
turn_right = 'turn_right_small_step_a'
turn_left  = 'turn_left_small_step_a'        
left_move = 'left_move_20'
right_move = 'right_move_20'
go_turn_right = 'turn_right'
go_turn_left = 'turn_left'

from hiwonder.CalibrationConfig import *    
# load parameters
param_data = np.load(calibration_param_path + '.npz')
mtx = param_data['mtx_array']
dist = param_data['dist_array']
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)

lab_data = None
servo_data = None
def load_config():
    global lab_data, servo_data
    
    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    servo_data = yaml_handle.get_yaml_data(yaml_handle.servo_file_path)

board = rrc.Board()
ctl = Controller(board)

# initial position
def initMove():
    ctl.set_pwm_servo_pulse(1,926,500)
    ctl.set_pwm_servo_pulse(2,servo_data['servo2'],500)   

object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
strp_up = True
# small state latch for detect/align/approach/commit-to-climb
stair_state = 'SEARCHING'

# consecutive good-detection counter, used to filter out
# noisy frames caused by the chest occluding the stair at close range
good_detection_count = 0
GOOD_DETECTION_REQUIRED = 3      # consecutive good frames required to commit
GOOD_WIDTH_MIN = 60              # min contour width (right_x-left_x); too narrow = occluded/too far
GOOD_ANGLE_MAX = 10              # max |angle|; occlusion makes angle jump to 10+
GOOD_LEFT_X_MIN = 20             # contour clipped against the left frame edge = occlusion artifact
CLOSE_COMMIT_Y = 400             # once center_y reaches this value the robot is already very close (just
                                  # below the 450 "still visible" cutoff, with some margin); if the contour is
                                  # no longer a "good detection" at that point (occlusion/perspective distortion),
                                  # stop trying to correct angle/x - it will only oscillate - and commit instead

# consecutive no-detection frame counter while SEARCHING/ALIGNING - prevents getting
# stuck forever, e.g. right after finishing a climb the next red line may momentarily
# be outside the camera's view because of the head angle/distance
no_detection_count = 0
NO_DETECTION_TIMEOUT = 30        # frames with no detection before trying to actively re-search
SEARCH_TILT_STEP = 40            # pan-tilt servo1 pulse adjustment per re-search attempt
SEARCH_TILT_MIN = 700            # lower bound for servo1 tilt, to avoid exceeding mechanical limits
SEARCH_TILT_MAX = 1100           # upper bound for servo1 tilt
search_tilt = 926                # current servo1 tilt used while searching, starts at the initial 926

# variable reset
def reset():
    global object_left_x, object_right_x
    global object_center_y, object_angle,strp_up
    global stair_state, good_detection_count
    
    global no_detection_count, search_tilt

    strp_up = True
    stair_state = 'SEARCHING'
    good_detection_count = 0
    no_detection_count = 0
    search_tilt = 926
    object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
    

# app initialization calling
def init():
    print("Stairway Init")
    load_config()
    initMove()
    AGC.runAction('stand_slow')

robot_is_running = False
# app start program calling
def start():
    global robot_is_running
    reset()
    robot_is_running = True
    print("Stairway Start")

# app stop program calling
def stop():
    global robot_is_running
    robot_is_running = False
    print("Stairway Stop")

# app exit program calling
def exit():
    global robot_is_running
    robot_is_running = False
    AGC.runActionGroup('stand_slow')
    print("Stairway Exit")


# find out the contour with the maximal area
# the list is the contour to be compared
def getAreaMaxContour(contours, area_min=10):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # iterate through all contours
        contour_area_temp = math.fabs(cv2.contourArea(c))  # calculate the contour area
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp >= area_min:  # only when the area is greater than the set value, the contour with the maximum area is considered valid to filter out interference
                area_max_contour = c

    return area_max_contour, contour_area_max  # return the contour with the maximal area

size = (640, 480)
# color block positioning vision processing function
def color_identify(img, img_draw, target_color = 'blue'):

    left_x, right_x, center_y, angle = -1, -1, -1, 0

    # occasionally the camera/image pipeline hands us an invalid frame -
    # zero-size, None, etc. Previously this raised straight out of here and
    # killed the caller's (vision thread's) loop, freezing the robot on the
    # last successfully detected values
    try:
        img_w = img.shape[:2][1]
        img_h = img.shape[:2][0]
        if img is None or img_w <= 0 or img_h <= 0:
            return left_x, right_x, center_y, angle

        img_resize = cv2.resize(img, (size[0], size[1]), interpolation = cv2.INTER_CUBIC)
        GaussianBlur_img = cv2.GaussianBlur(img_resize, (3, 3), 3)# Gaussian blur
        frame_lab = cv2.cvtColor(GaussianBlur_img, cv2.COLOR_BGR2LAB) # convert the image to LAB space
        frame_mask = cv2.inRange(frame_lab,
                                     (lab_data[target_color]['min'][0],
                                      lab_data[target_color]['min'][1],
                                      lab_data[target_color]['min'][2]),
                                     (lab_data[target_color]['max'][0],
                                      lab_data[target_color]['max'][1],
                                      lab_data[target_color]['max'][2]))  # operate bitwise operation to original image and mask
        opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))# opening operation
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))# closing operation
        contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2] # find out all the bounding contours
        areaMax_contour = getAreaMaxContour(contours, area_min=50)[0] # find out the contour with the maximal area
    except Exception as e:
        print('color_identify error:', e)
        return -1, -1, -1, 0

    if areaMax_contour is not None:
        
        down_x = (areaMax_contour[areaMax_contour[:,:,1].argmax()][0])[0]
        down_y = (areaMax_contour[areaMax_contour[:,:,1].argmax()][0])[1]

        left_x = (areaMax_contour[areaMax_contour[:,:,0].argmin()][0])[0]
        left_y = (areaMax_contour[areaMax_contour[:,:,0].argmin()][0])[1]

        right_x = (areaMax_contour[areaMax_contour[:,:,0].argmax()][0])[0]
        right_y = (areaMax_contour[areaMax_contour[:,:,0].argmax()][0])[1]
        
        if pow(down_x - left_x, 2) + pow(down_y - left_y, 2) > pow(down_x - right_x, 2) + pow(down_y - right_y, 2):
            right_x = down_x
            right_y = down_y
        else:
            left_x = down_x
            left_y = down_y

        center_y = down_y
        angle = int(math.degrees(math.atan2(right_y - left_y, right_x - left_x)))
        
        cv2.line(img_draw, (left_x, left_y), (right_x, right_y), (255, 0, 0), 2)
            
    return left_x, right_x, center_y, angle      


# execute the final climb/descend action sequence; once called it no longer relies on vision
def do_climb():
    global strp_up, object_center_y, stair_state, good_detection_count

    print('STATE: COMMITTED_TO_CLIMB')
    stair_state = 'COMMITTED_TO_CLIMB'
    good_detection_count = 0
    time.sleep(0.8)

    print('STATE: CLIMBING')
    stair_state = 'CLIMBING'
    board.set_buzzer(1900, 0.1, 0.9, 1)
    for i in range(2):
        AGC.runActionGroup(go_forward_one_small_step) # take a small step forward
        time.sleep(0.5)

    if strp_up: # go up stair
        AGC.runActionGroup('climb_stairs')
        strp_up = False
    else:       # go down stair
        for i in range(2):
            AGC.runActionGroup(go_forward_one_small_step) # take a small step forward
        time.sleep(0.5)
        AGC.runActionGroup('down_floor')
        strp_up = True
    time.sleep(0.5)

    # action groups such as climb_stairs/down_floor drive every servo, including the
    # pan-tilt servo1. They do not restore it afterwards - it's simply left wherever the
    # action group's last frame put it, typically well below 926 - so the camera ends up
    # tilted at the wrong angle and can no longer see the next red line for the next stair
    initMove()
    time.sleep(0.2)
    object_center_y = -1

    print('STATE: DONE')
    stair_state = 'SEARCHING'


# robot tracking thread
def move():
    global strp_up
    global object_center_y
    global stair_state
    global good_detection_count
    global no_detection_count, search_tilt
    
    centreX = 320 # the pixel coordinates of the object corresponding to the center point directly in front of the robot may not align with the actual center of the object due to installation errors
    
    while True:
        if robot_is_running:
            if stair_state in ('COMMITTED_TO_CLIMB', 'CLIMBING'):
                # already committed to climb, ignore any loss of vision until it finishes
                time.sleep(0.01)
            elif object_center_y >= 0:  # detected stair, perform positional fine-tuning
                no_detection_count = 0
                object_x = object_left_x + (object_right_x - object_left_x)/2
                object_width = object_right_x - object_left_x

                # good-detection filter: sufficient width, reasonable angle, contour not
                # clipped against the left frame edge - distinguishes a real stair contour
                # from a noisy frame caused by chest occlusion
                good_detection = (object_width >= GOOD_WIDTH_MIN and
                                   abs(object_angle) <= GOOD_ANGLE_MAX and
                                   object_left_x >= GOOD_LEFT_X_MIN)
                if good_detection:
                    good_detection_count += 1
                else:
                    good_detection_count = 0

                print("STAIR:", "state=", stair_state, "width=", object_width,
                      "center_y=", object_center_y, "angle=", object_angle,
                      "good=", good_detection, "good_count=", good_detection_count)

                if good_detection_count >= GOOD_DETECTION_REQUIRED:
                    # several consecutive stable, wide, well-angled detections -> close and
                    # aligned enough; ignore this frame's angle/x-offset - which may be
                    # distorted by occlusion - and commit to the climb sequence
                    do_climb()
                    continue

                if (not good_detection and object_center_y >= CLOSE_COMMIT_Y
                        and abs(object_x - centreX) < 150 and stair_state != 'SEARCHING'):
                    # already very close (high center_y) AND roughly centered left/right, but the
                    # contour is narrowed/angle-distorted by perspective/occlusion - same class of
                    # problem as losing vision while APPROACHING. Continuing to run angle/x correction
                    # branches would just make the robot oscillate in place forever since the distortion
                    # isn't a real offset, so commit to the climb sequence directly.
                    # The alignment check (abs(object_x - centreX) < 150) is required here: right after
                    # finishing an "up" climb, the *next* (descend) red line can appear with a high
                    # center_y on the very first frame (different camera perspective looking down from
                    # the top step) while still being badly off-center - committing immediately in that
                    # case skips lateral alignment entirely and the robot fails to actually line up with
                    # the down-stair before triggering down_floor.
                    # The stair_state != 'SEARCHING' guard is required too: right after do_climb()
                    # resets to SEARCHING, the very first detected frame of the *next* stair can
                    # already report a high center_y (perspective from the top step) while ALSO
                    # happening to be roughly x-centered, even though its angle is genuinely off
                    # (e.g. angle=14, a real turn is needed, not occlusion noise). Requiring at least
                    # one prior pass through the normal ALIGNING/APPROACHING branches for this stair
                    # ensures a real angle offset gets corrected (via the 3<angle<20 / -20<angle<-5
                    # branches) before this shortcut is allowed to bypass it and commit too early.
                    do_climb()
                    continue

                if object_center_y < 320 and abs(object_x - centreX) < 150:  # approach quickly
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(go_forward)
                    time.sleep(0.2)
                
                elif 20 <= object_angle < 90:  # angle adjustment
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(go_turn_right)
                    time.sleep(0.2)           
                elif -20 >= object_angle > -90:
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(go_turn_left)
                    time.sleep(0.2)
                    
                elif object_x - centreX > 15: # adjust left and right
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(right_move)
                elif object_x - centreX < -15:
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(left_move)
                
                elif 3 < object_angle < 20:   # adjust the angle slightly
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(turn_right)
                    time.sleep(0.2)           
                elif -5 > object_angle > -20:
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(turn_left)
                    time.sleep(0.2)
                    
                elif 320 <= object_center_y < 450:   # centered, aligned with known distance -> approaching
                    if stair_state != 'APPROACHING':
                        print('STATE: APPROACHING')
                        stair_state = 'APPROACHING'
                    AGC.runActionGroup(go_forward_one_step)
                    time.sleep(0.2)
                    
                elif object_center_y >= 450: # close enough and still visible, commit to climb
                    do_climb()
                    
                else:
                    time.sleep(0.01)
            elif stair_state == 'APPROACHING':
                # already aligned and approaching; vision lost here is treated as chest occlusion at close range, so commit anyway
                do_climb()
            else:
                # SEARCHING/ALIGNING with no detection at all - e.g. right after finishing a
                # climb, the next red line may momentarily be outside the camera's view because
                # of the head tilt/distance. Rather than freezing forever, actively nudge the
                # head tilt and take a small step to try to re-acquire the stair
                no_detection_count += 1
                if no_detection_count >= NO_DETECTION_TIMEOUT:
                    no_detection_count = 0
                    search_tilt += SEARCH_TILT_STEP
                    if search_tilt > SEARCH_TILT_MAX or search_tilt < SEARCH_TILT_MIN:
                        search_tilt = 926
                    print('STAIR: re-search, tilt=', search_tilt)
                    ctl.set_pwm_servo_pulse(1, search_tilt, 300)
                    time.sleep(0.3)
                    AGC.runActionGroup(go_forward_one_small_step)
                    time.sleep(0.5)
                else:
                    time.sleep(0.01)
        else:
            time.sleep(0.01)
                
            
# start as a sub-thread
th = threading.Thread(target=move)
th.daemon = True 
th.start()


def run(img):
    global object_left_x, object_right_x
    global object_center_y, object_angle

    img_copy = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    if not robot_is_running:
        return img_copy
    
    # go up and down stair
    object_left_x, object_right_x, object_center_y, object_angle = color_identify(img_copy.copy(), img_copy, target_color = 'red')
    # print('stairway',object_left_x, object_right_x, object_center_y, object_angle)# print position and angle parameters
            
        
    return img_copy

if __name__ == '__main__':
    
    my_camera = Camera.Camera()
    my_camera.camera_open()
    
    init()
    start()
    
    while True:
        ret,img = my_camera.read()
        if ret:
            frame = img.copy()
            Frame = run(frame)           
            cv2.imshow('Frame', Frame)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            time.sleep(0.01)
    my_camera.camera_close()
    cv2.destroyAllWindows()

