import json
import rospy
from std_msgs.msg import String
from nao_alproxy import NaoALProxy
import sys
import time


class NaoListenerNodeMultiple():
    # def __init__(self, nao_ip=['192.168.0.100', '192.168.0.102']):
    # def __init__(self, nao_ip=['192.168.0.100']):
    def __init__(self, nao_ip=[]):
        # input is an array of ip's (in strings)
        self.nao_alproxy = []
        self.publisher = []
        for i, ip in enumerate(nao_ip):
            self.nao_alproxy.append(NaoALProxy(ip))
            self.nao_alproxy[-1].start_nao()
            self.publisher.append(rospy.Publisher('nao_state_%d' % i, String, queue_size=10))
            rospy.Subscriber('to_nao_%d' % i, String, self.callback_to_nao, callback_args=i)

        rospy.init_node('nao_listener_node')  # init a listener:
        # rospy.Subscriber('nao_state', String, self.callback_nao_state)
        print('=========== NaoListenerNode =============')

        rospy.spin()  # spin() simply keeps python from exiting until this node is stopped

    def callback_to_nao(self, data, i):
        print("callback_robotator", data.data)
        message = data.data

        rospy.loginfo(message)
        self.nao_alproxy[i].parse_message(message)
        # print("finished", message)
        self.publisher[i].publish(data.data)  # publish to nao_state to indicate that the robot command is complete

    # def callback_nao_state (self, data):
    #    print("nao_ros_listener callback_nao_state", data.data)


if __name__ == '__main__':
    try:
        # from command line: first input is an optional nao_ip
        if len(sys.argv) > 1:
            nao = NaoListenerNodeMultiple(sys.argv[1:])
        else:
            nao = NaoListenerNodeMultiple()
    except rospy.ROSInterruptException:
        pass