#!/usr/bin/python3
# coding=utf8
# 4.拓展课程学习\11.拓展课程之田径运动课程\第2课 爬台阶(4.Advanced Lessons\11.Athletics Sport Lesson\Lesson2 Go Up and Down Stair)
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

# 上下台阶(go up and down stair)

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
#加载参数(load parameters)
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

# 初始位置(initial position)
def initMove():
    ctl.set_pwm_servo_pulse(1,926,500)
    ctl.set_pwm_servo_pulse(2,servo_data['servo2'],500)   

object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
strp_up = True
# 阶梯识别/对齐/接近/锁定爬阶的小状态机(small state latch for detect/align/approach/commit-to-climb)
stair_state = 'SEARCHING'

# 连续“良好检测”计数,用于抵御胸部遮挡造成的噪声帧(consecutive good-detection counter, used to filter out
# noisy frames caused by the chest occluding the stair at close range)
good_detection_count = 0
GOOD_DETECTION_REQUIRED = 3      # 需要连续多少帧良好检测才允许锁定爬阶(consecutive good frames required to commit)
GOOD_WIDTH_MIN = 60              # 轮廓宽度(right_x-left_x),太窄说明台阶被遮挡/太远(min contour width; too narrow = occluded/too far)
GOOD_ANGLE_MAX = 10              # 角度绝对值上限,遮挡时角度会突然跳到10+度(max |angle|; occlusion makes angle jump to 10+)
GOOD_LEFT_X_MIN = 20             # 轮廓左边界离画面左边缘太近，说明轮廓被裁切(遮挡的特征)(contour clipped against the left frame edge = occlusion artifact)

# 变量重置(variable reset)
def reset():
    global object_left_x, object_right_x
    global object_center_y, object_angle,strp_up
    global stair_state, good_detection_count
    
    strp_up = True
    stair_state = 'SEARCHING'
    good_detection_count = 0
    object_left_x, object_right_x, object_center_y, object_angle = -2, -2, -2, 0
    

# app初始化调用(app initialization calling)
def init():
    print("Stairway Init")
    load_config()
    initMove()
    AGC.runAction('stand_slow')

robot_is_running = False
# app开始玩法调用(app start program calling)
def start():
    global robot_is_running
    reset()
    robot_is_running = True
    print("Stairway Start")

# app停止玩法调用(app stop program calling)
def stop():
    global robot_is_running
    robot_is_running = False
    print("Stairway Stop")

# app退出玩法调用(app exit program calling)
def exit():
    global robot_is_running
    robot_is_running = False
    AGC.runActionGroup('stand_slow')
    print("Stairway Exit")


# 找出面积最大的轮廓(find out the contour with the maximal area)
# 参数为要比较的轮廓的列表(the list is the contour to be compared)
def getAreaMaxContour(contours, area_min=10):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # 历遍所有轮廓(iterate through all contours)
        contour_area_temp = math.fabs(cv2.contourArea(c))  # 计算轮廓面积(calculate the contour area)
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp >= area_min:  # 只有在面积大于设定值时，最大面积的轮廓才是有效的，以过滤干扰(only when the area is greater than the set value, the contour with the maximum area is considered valid to filter out interference)
                area_max_contour = c

    return area_max_contour, contour_area_max  # 返回最大的轮廓(return the contour with the maximal area)

size = (640, 480)
# 色块定位视觉处理函数(color block positioning vision processing function)
def color_identify(img, img_draw, target_color = 'blue'):
    
    img_w = img.shape[:2][1]
    img_h = img.shape[:2][0]
    img_resize = cv2.resize(img, (size[0], size[1]), interpolation = cv2.INTER_CUBIC)
    GaussianBlur_img = cv2.GaussianBlur(img_resize, (3, 3), 3)#高斯模糊(Gaussian blur)
    frame_lab = cv2.cvtColor(GaussianBlur_img, cv2.COLOR_BGR2LAB) #将图像转换到LAB空间(convert the image to LAB space)
    frame_mask = cv2.inRange(frame_lab,
                                 (lab_data[target_color]['min'][0],
                                  lab_data[target_color]['min'][1],
                                  lab_data[target_color]['min'][2]),
                                 (lab_data[target_color]['max'][0],
                                  lab_data[target_color]['max'][1],
                                  lab_data[target_color]['max'][2]))  #对原图像和掩模进行位运算(operate bitwise operation to original image and mask)
    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))#开运算(opening operation)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))#闭运算(closing operation)
    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2] #找出所有外轮廓(find out all the bounding contours)
    areaMax_contour = getAreaMaxContour(contours, area_min=50)[0] #找到最大的轮廓(find out the contour with the maximal area)

    left_x, right_x, center_y, angle = -1, -1, -1, 0
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


# 执行最终爬阶/下台阶动作序列，一旦调用不再依赖视觉(execute the final climb/descend action sequence; once called it no longer relies on vision)
def do_climb():
    global strp_up, object_center_y, stair_state

    print('STATE: COMMITTED_TO_CLIMB')
    stair_state = 'COMMITTED_TO_CLIMB'
    time.sleep(0.8)

    print('STATE: CLIMBING')
    stair_state = 'CLIMBING'
    board.set_buzzer(1900, 0.1, 0.9, 1)
    for i in range(2):
        AGC.runActionGroup(go_forward_one_small_step) #前进一小步(take a small step forward)
        time.sleep(0.5)

    if strp_up: # 上台阶(go up stair)
        AGC.runActionGroup('climb_stairs')
        strp_up = False
    else:       # 下台阶(go down stair)
        for i in range(2):
            AGC.runActionGroup(go_forward_one_small_step) #前进一步(take a small step forward)
        time.sleep(0.5)
        AGC.runActionGroup('down_floor')
        strp_up = True
    time.sleep(0.5)
    object_center_y = -1

    print('STATE: DONE')
    stair_state = 'SEARCHING'


#机器人跟踪线程(robot tracking thread)
def move():
    global strp_up
    global object_center_y
    global stair_state
    global good_detection_count
    
    centreX = 320 # 物体在机器人正前方中心点对应的像素坐标,由于安装误差，物体在画面中心并不对应物体就在机器人中心点(the pixel coordinates of the object corresponding to the center point directly in front of the robot may not align with the actual center of the object due to installation errors)
    
    while True:
        if robot_is_running:
            if stair_state in ('COMMITTED_TO_CLIMB', 'CLIMBING'):
                # 已锁定爬阶，不再理会视觉丢失(already committed to climb, ignore any loss of vision until it finishes)
                time.sleep(0.01)
            elif object_center_y >= 0:  #检测到台阶,进行位置微调(detected stair, perform positional fine-tuning)
                object_x = object_left_x + (object_right_x - object_left_x)/2
                object_width = object_right_x - object_left_x

                # “良好检测”过滤:宽度足够、角度合理、轮廓未被画面左边缘裁切,
                # 用来区分真实台阶轮廓和胸部遮挡产生的噪声帧
                # (good-detection filter: sufficient width, reasonable angle, contour not
                # clipped against the left frame edge - distinguishes a real stair contour
                # from a noisy frame caused by chest occlusion)
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
                    # 已连续多帧看到稳定、宽阔、角度良好的台阶轮廓,判定已足够接近且对齐,
                    # 不再理会本帧(可能因遮挡而失真)的角度/左右偏移，直接锁定爬阶
                    # (several consecutive stable, wide, well-angled detections -> close and
                    # aligned enough; ignore this frame's angle/x-offset - which may be
                    # distorted by occlusion - and commit to the climb sequence)
                    do_climb()
                    good_detection_count = 0
                    continue

                if object_center_y < 320 and abs(object_x - centreX) < 150:  #快速靠近(approach quickly)
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(go_forward)
                    time.sleep(0.2)
                
                elif 20 <= object_angle < 90:  #角度调整(angle adjustment)
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
                    
                elif object_x - centreX > 15: #左右调整(adjust left and right)
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(right_move)
                elif object_x - centreX < -15:
                    if stair_state != 'ALIGNING':
                        print('STATE: ALIGNING')
                        stair_state = 'ALIGNING'
                    AGC.runActionGroup(left_move)
                
                elif 3 < object_angle < 20:   #角度微调(adjust the angle slightly)
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
                    
                elif 320 <= object_center_y < 450:   #在中心，已对齐且距离已知，进入接近状态(centered, aligned with known distance -> approaching)
                    if stair_state != 'APPROACHING':
                        print('STATE: APPROACHING')
                        stair_state = 'APPROACHING'
                    AGC.runActionGroup(go_forward_one_step)
                    time.sleep(0.2)
                    
                elif object_center_y >= 450: #位置靠近，仍能看见台阶，直接锁定爬阶(close enough and still visible, commit to climb)
                    do_climb()
                    
                else:
                    time.sleep(0.01)
            elif stair_state == 'APPROACHING':
                # 已对齐并处于接近状态，此时台阶因胸部遮挡而消失，视为已到达可爬阶位置，直接锁定(already aligned and approaching; vision lost here is treated as chest occlusion at close range, so commit anyway)
                do_climb()
            else:
                time.sleep(0.01)
        else:
            time.sleep(0.01)
                
            
#作为子线程开启(start as a sub-thread)
th = threading.Thread(target=move)
th.daemon = True 
th.start()


def run(img):
    global object_left_x, object_right_x
    global object_center_y, object_angle

    img_copy = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    if not robot_is_running:
        return img_copy
    
    # 上下台阶(go up and down stair)
    object_left_x, object_right_x, object_center_y, object_angle = color_identify(img_copy.copy(), img_copy, target_color = 'red')
    print('stairway',object_left_x, object_right_x, object_center_y, object_angle)# 打印位置角度参数
            
        
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

