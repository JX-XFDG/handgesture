import cv2
import mediapipe as mp
import pyautogui
import keyboard
import time
from dotdistance import dotdistance 
from datetime import datetime
import pygame 
mloptwl_t=0 # 比例初始化
mlortwl_t=0 # 比例初始化
mlomtwl_t=0 # 比例初始化
mloptwl=0
mlortwl=0
mlomtwl=0
stopflag = 0# 停止标志位初始化
timestamps = []
class HandTracker:
    def __init__(self, 
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        初始化手部追踪器

        static_image_mode 视频流模式
        max_num_hands 最大检测手数量
        min_detection_confidence 检测置信度阈值
        min_tracking_confidence 跟踪置信度阈值
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 初始化手部检测模型
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # 视频捕获对象
        self.cap = None
        self.frame = None
        
        # 样式配置
        self.landmark_style = self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)# 关键点样式：红色、2像素粗细、2像素半径
        self.connection_style = self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)# 连接线样式：绿色、2像素粗细
    def showtracker(self, results, image): # 添加参数传递
        '''显示骨骼'''
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,  # 使用传入的image
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.landmark_style,
                    self.connection_style
                )
    def getfingerlocate(self, results, image, finger,hand=None):
        '''获取指定手指的坐标
        results 手部检测结果
        image 图像
        finger 指定哪一个手指
        hand 指定哪只手
        finger 参数表：
        WRIST	0	手腕中心点	-
        THUMB_CMC	1	拇指根部（掌指关节）	拇指
        THUMB_MCP	2	拇指近端指节	拇指
        THUMB_IP	3	拇指中间指节	拇指
        THUMB_TIP	4	拇指指尖	拇指
        INDEX_FINGER_MCP	5	食指近端指节	食指
        INDEX_FINGER_PIP	6	食指中间指节	食指
        INDEX_FINGER_DIP	7	食指远端指节	食指
        INDEX_FINGER_TIP	8	食指指尖	食指
        MIDDLE_FINGER_MCP	9	中指近端指节	中指
        MIDDLE_FINGER_PIP	10	中指中间指节	中指
        MIDDLE_FINGER_DIP	11	中指远端指节	中指
        MIDDLE_FINGER_TIP	12	中指指尖	中指
        RING_FINGER_MCP	13	无名指近端指节	无名指
        RING_FINGER_PIP	14	无名指中间指节	无名指
        RING_FINGER_DIP	15	无名指远端指节	无名指
        RING_FINGER_TIP	16	无名指指尖	无名指
        PINKY_MCP	17	小指近端指节	小指
        PINKY_PIP	18	小指中间指节	小指
        PINKY_DIP	19	小指远端指节	小指
        PINKY_TIP	20	小指指尖	小指
        '''
        height, width, _ = image.shape
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 获取手部方向信息
                handedness = results.multi_handedness[hand_idx].classification[0].label
                # 筛选指定的手（Left/Right），未指定时处理所有手
                if hand and handedness != hand:
                    continue
                # 获取食指指尖坐标（INDEX_FINGER_TIP = 8）
                index_finger_tip = hand_landmarks.landmark[finger]
                # 转换归一化坐标到像素坐标
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)
                # 在指尖画红色圆圈
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                # 显示手方向标签（自动根据实际方向显示）
                # cv2.putText(image, f"{handedness} Hand", (x, y-20),
                            # cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # 显示坐标文本
                cv2.putText(image, f"Index: ({x}, {y})", (x+15, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return x,y
        return None, None
    def mousemove(self,x,y,screen_w,screen_h,endx,endy,shakeproof):
        '''鼠标移动
        x,y 现在手指坐标值
       screen_w,screen_h, 屏幕大小 
       endx,endy 用来计算手指坐标值差值的藏书
       shakeproof 防抖参数'''
        prev_x, prev_y = pyautogui.position()
        smooth_factor = 0.8# 平滑移动参数（可选）
        if x is not None and y is not None and abs(endx-x) > shakeproof*3 and abs(endy - y) > shakeproof*2:
            # 坐标映射到屏幕
            screen_x = int((x / 640) * screen_w)
            screen_y = int((y / 480) * screen_h)
            # 平滑移动（减少抖动）
            smoothed_x = prev_x + (screen_x - prev_x) * smooth_factor
            smoothed_y = prev_y + (screen_y - prev_y) * smooth_factor
            # 移动鼠标
            pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)
            prev_x, prev_y = smoothed_x, smoothed_y

dandtracker = HandTracker()
screen_w, screen_h = pyautogui.size() # 获取屏幕分辨率
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # 启用DirectShow加速
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 设置视频宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置视频高度
start_time = time.time()# 获取初始时间
flag_once = 0# 用来执行那些只需要计算一次的东西的标志位
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = dandtracker.hands.process(rgb_frame)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    end_time = time.time()
    dandtracker.showtracker(results, bgr_frame)
    '''获取手指坐标'''
    x_index_tip,y_index_tip = dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.INDEX_FINGER_TIP,'Right')
    x_wrist_tip,y_wrist_tip = dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.WRIST,'Right')
    dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.THUMB_TIP,'Right    ')
    x_middle_tip,y_middle_tip = dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,'Right')
    x_middle_mcp,y_middle_mcp =dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,'Right')
    x_middle_mcp,y_middle_mcp =dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,'Right')
    x_rinky_tip,y_rinky_tip = dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.PINKY_TIP,'Right')# 小指指尖坐标
    x_ring_tip,y_ring_tip = dandtracker.getfingerlocate(results,bgr_frame,dandtracker.mp_hands.HandLandmark.RING_FINGER_TIP,'Right')# 无名指指尖坐标

    '''实时计算'''
    try: # 防止手不在摄像头里没有目标而报错
        '''计算实时长度'''
        middlefingerlong_t = dotdistance(x_middle_tip,y_middle_tip,x_middle_mcp,y_middle_mcp)# 中指的长度
        pinkytowaistlong_t = dotdistance(x_wrist_tip,y_wrist_tip,x_rinky_tip,y_rinky_tip)# 小指到手腕的长度
        ringtowaistlong_t = dotdistance(x_wrist_tip,y_wrist_tip,x_ring_tip,y_ring_tip)# 无名指到手腕的长度
        midtowaistlong_t = dotdistance(x_middle_tip,y_middle_tip,x_wrist_tip,y_wrist_tip)# 中指到手腕的长度
        '''计算实时比例'''
        mloptwl_t = middlefingerlong_t/pinkytowaistlong_t
        mlortwl_t = middlefingerlong_t/ringtowaistlong_t
        mlomtwl_t = middlefingerlong_t/midtowaistlong_t
    except:
        pinkytowaistlong_t = 1
        ringtowaistlong_t = 2
        middlefingerlong_t = 3
    if flag_once ==0:
        if keyboard.is_pressed('1'):
            print('进入键盘为1')
            '''计算长度'''
            middlefingerlong = dotdistance(x_middle_tip,y_middle_tip,x_middle_mcp,y_middle_mcp)# 中指的长度
            pinkytowaistlong = dotdistance(x_wrist_tip,y_wrist_tip,x_rinky_tip,y_rinky_tip)# 小指到手腕的长度
            ringtowaistlong = dotdistance(x_wrist_tip,y_wrist_tip,x_ring_tip,y_ring_tip)# 无名指到手腕的长度
            midtowaistlong = dotdistance(x_middle_tip,y_middle_tip,x_wrist_tip,y_wrist_tip)# 中指到手腕的长度
            '''计算比例'''
            mloptwl = middlefingerlong/pinkytowaistlong
            mlortwl = middlefingerlong/ringtowaistlong
            mlomtwl = middlefingerlong/midtowaistlong

    '''判断手掌打开'''
    # print(f'{abs(mloptwl_t-mloptwl)},{abs(mlortwl_t-mlortwl)},{abs(mlomtwl_t-mlomtwl)}')
    print(timestamps)
    if mloptwl != 0 and mlortwl != 0 and mlomtwl != 0:
        # 检查误差是否在 0.04 以内
        if abs(mloptwl_t - mloptwl) < 0.04 and abs(mlortwl_t - mlortwl) < 0.04 and abs(mlomtwl_t - mlomtwl) < 0.04:
            current_time = time.time()
    
            # **优化部分**: 直接用列表推导式删除超过2秒的数据
            timestamps = [t for t in timestamps if t >= current_time - 2]
            
            timestamps.append(current_time)  # 记录当前时间戳
            
            # **如果 2 秒内记录达到了 30 次**
            if len(timestamps) >= 30:
                print('执行')
                stopflag ^= 1  # 切换暂停/开始状态
                timestamps.clear()  # **触发后清空列表**
    if(stopflag == 1):
        cv2.putText(bgr_frame, "STOP", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    '''鼠标移动'''
    if(stopflag != 1):
        if(end_time - start_time) > 1:# 计算防抖参数
            if x_index_tip != None and y_index_tip != None:
                endx = x_index_tip
                endy = y_index_tip
            start_time = end_time
            flag_mousemove = 1
        try:
            dandtracker.mousemove(x_index_tip,y_index_tip,screen_w,screen_h,endx,endy,1)
        except:# 防止刚进程序手不在摄像头里导致endx，endy没有被赋值而报错
            endx = screen_w/2
            endy = screen_h/2
            continue
    cv2.imshow('Hand Tracking', bgr_frame)
    if cv2.waitKey(5) & 0xFF == 27: # 按ESC退出
        break
