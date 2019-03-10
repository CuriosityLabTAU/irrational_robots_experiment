import json
import rospy
from std_msgs.msg import String
import sys
import time
from threading import Timer
import threading
import random

import subprocess

import time
import naoqi




# todo: figure how to this automatically in the background
# import twisted_server_ros
# import nao_ros_listener_multiple
#
# twisted_server_ros.TwistedServerApp().run()
# try:
#     # from command line: first input is an optional nao_ip
#     if len(sys.argv) > 1:
#         nao = nao_ros_listener_multiple.NaoListenerNodeMultiple(sys.argv[1:])
#     else:
#         nao = nao_ros_listener_multiple.NaoListenerNodeMultiple()
# except rospy.ROSInterruptException:
#     pass

# subprocess.Popen(['python','nao_comm/twisted_server_ros.py'], shell=False)
# time.sleep(5)
#
# subprocess.Popen(['python','nao_comm/nao_ros_listener_multiple.py'], shell=False)
# time.sleep(5)


robot_sound_path = '/home/nao/irrational/sounds/'
# robot_behavior_path = 'facilitator-6ea3b8/'
robot_behavior_path = 'torr_test_v1-3714cd/'


def run_robot_behavior(robots_publisher, which_robot, message):
    robots_publisher[which_robot].publish(json.dumps(message))
    # waiting_robot = True
    # while waiting_robot:
    #     pass
    # print('action ended')

def callback_nao_state(data):
    if 'register tablet' not in data.data and 'sound_tracker' not in data.data:
        waiting = False
        waiting_robot = False

        try:
            signal = json.loads(data.data)['parameters'][0]
            # robot_end_signal[signal] = True
        except:
            pass

robots_publisher = []
for i in range(2):
    robots_publisher.append(rospy.Publisher('to_nao_%d' % i, String, queue_size=10))
    rospy.init_node('manager_node')  # init a listener:
    rospy.Subscriber('nao_state_%d' % i, String, callback_nao_state)

# how to run an action on the robot
# action = {"action": "wake_up"}
action = {'action': 'run_behavior','parameters': [robot_behavior_path+'suspect_a', 'wait']}
run_robot_behavior(robots_publisher, 0, action)

## how to play a file
#
# for r in range(2):
#     for percent in [40, 60, 80]:
#         the_action = robot_sound_path + '%d.wav' % percent
#         nao_message = {"action": 'play_audio_file',
#                        "parameters": [the_action]}
#         run_robot_behavior(robots_publisher, r, nao_message)
#         time.sleep(2)
#     time.sleep(3)

### browse/copy files to nao through the terminal
# ssh nao@192.168.0.100
# pass: nao
# linux operations
# copy content of one folder to another:
# scp -r /home/curious/PycharmProjects/irrational_robots_experiment/sounds nao@192.168.0.100:/home/nao/irrational

# ./.local/share/PackageManager/apps/irrational
# naoqi.ALBehavior()

# todo: multiple robots

