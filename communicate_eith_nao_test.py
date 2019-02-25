import json
import rospy
from std_msgs.msg import String
import time
from threading import Timer
import threading
import random


robot_sound_path = '/home/nao/irrational/sounds/'
robot_behavior_path = 'facilitator-6ea3b8/'


def run_robot_behavior(nao_message):
    robot_publisher.publish(json.dumps(nao_message))

def callback_nao_state(data):
    if 'register tablet' not in data.data and 'sound_tracker' not in data.data:
        waiting = False
        waiting_robot = False

        try:
            signal = json.loads(data.data)['parameters'][0]
            # robot_end_signal[signal] = True
        except:
            pass

robot_publisher = rospy.Publisher('to_nao', String, queue_size=10)
rospy.init_node('manager_node')  # init a listener:
rospy.Subscriber('nao_state', String, callback_nao_state)

# how to run an action on the robot
# action = {"action": "wake_up"}
action = {'action': 'run_behavior','parameters': ['robot_facilitator-ad2c5c/robotator_behaviors/TU13b', 'wait']}
run_robot_behavior(action)

### how to play a file
the_action = robot_sound_path + '90.wav'
nao_message = {"action": 'play_audio_file',
               "parameters": [the_action]}
run_robot_behavior(nao_message)

### browse/copy files to nao through the terminal
# ssh nao@192.168.0.100
# pass: nao
# linux operations
# copy content of one folder to another:
# scp -r /home/curious/PycharmProjects/irrational_robots_experiment/sounds nao@192.168.0.100:/home/nao/irrational

# todo: multiple robots

