import math
from loguru import logger

def angle_between_points( p0, p1, p2 ):
    # 计算角度
    a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
    b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
    if a * b == 0:
        return -1.0
    return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180 /math.pi


def length_between_points(p0, p1):
    # 2点之间的距离
    return math.hypot(p1[0]- p0[0], p1[1]-p0[1])

def get_angle_point(human, pos):
    # 返回各个部位的关键点
    pnts = []
    if pos == 'left_elbow':
        #pos_list = (5,6,7)
        pos_list = [2, 3, 4]
    elif pos == 'left_hand':
        #pos_list = (1,5,7)
        pos_list = [1, 2, 4]#(1, 2, 4)
    elif pos == 'left_knee':
        #pos_list = (12,13,14)
        pos_list = [8, 9, 10]#(8, 9, 10)
    elif pos == 'left_ankle':
        #pos_list = (5,12,14)
        pos_list = [2,8,10]#(2,8,10)
    elif pos == 'right_elbow':
        #pos_list = (2,3,4)
        pos_list = [5, 6, 7]#(5, 6, 7)
    elif pos == 'right_hand':
        #pos_list = (1,2,4)
        pos_list = [1,5,7]#(1, 5, 7)
    elif pos == 'right_knee':
        #pos_list = (9,10,11)
        pos_list = [11,12,13]#(11,12, 13)
    elif pos == 'right_ankle':
        #pos_list = (2,9,11)
        pos_list = [5, 11, 13]#(5, 11, 13)
    else:
        print('Unknown  [%s]', pos)
        return pnts

    for i in range(3):
        #print("pos_list[i]:",pos_list[i])
        #print("human:",human)
        for j in range(len(human)):#range(17):
            if human[j][-1] == pos_list[i]:
                pnts.append((int(human[j][0]),int(human[j][1])))
            
    return pnts


def angle_left_hand(human):
    pnts = get_angle_point(human, 'left_hand')
    if len(pnts) != 3:
        print('left_hand component incomplete')
        logger.info('left_hand component incomplete')
        return -1
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left hand angle:%f'%(angle))
        logger.info('left hand angle:%f'%(angle))
    return angle


def angle_left_elbow(human):
    pnts = get_angle_point(human, 'left_elbow')
    if len(pnts) != 3:
        print('left_elbow component incomplete')
        logger.info('left_elbow component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left elbow angle:%f'%(angle))
        logger.info('left elbow angle:%f'%(angle))
    return angle


def angle_left_knee(human):
    pnts = get_angle_point(human, 'left_knee')
    if len(pnts) != 3:
        print('left_knee component incomplete')
        logger.info('left_knee component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left knee angle:%f'%(angle))
        logger.info('left knee angle:%f'%(angle))
    return angle


def angle_left_ankle(human):
    pnts = get_angle_point(human, 'left_ankle')
    if len(pnts) != 3:
        print('left_ankle component incomplete')
        logger.info('left_ankle component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('left ankle angle:%f'%(angle))
        logger.info('left ankle angle:%f'%(angle))
    return angle


def angle_right_hand(human):
    pnts = get_angle_point(human, 'right_hand')
    if len(pnts) != 3:
        print('right_hand component incomplete')
        logger.info('right_hand component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right hand angle:%f'%(angle))
        logger.info('right hand angle:%f'%(angle))
    return angle


def angle_right_elbow(human):
    pnts = get_angle_point(human, 'right_elbow')
    if len(pnts) != 3:
        print('right_elbow component incomplete')
        logger.info('right_elbow component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right elbow angle:%f'%(angle))
        logger.info('right elbow angle:%f'%(angle))
    return angle


def angle_right_knee(human):
    pnts = get_angle_point(human, 'right_knee')
    if len(pnts) != 3:
        print('right_knee component incomplete')
        logger.info('right_knee component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right knee angle:%f'%(angle))
        logger.info('right knee angle:%f'%(angle))
    return angle


def angle_right_ankle(human):
    pnts = get_angle_point(human, 'right_ankle')
    if len(pnts) != 3:
        print('right_ankle component incomplete')
        logger.info('right_ankle component incomplete')
        return
    angle = 0
    if pnts is not None:
        angle = angle_between_points(pnts[0], pnts[1], pnts[2])
        print('right ankle angle:%f'%(angle))
        logger.info('right ankle angle:%f'%(angle))
    return angle
