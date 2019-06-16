import json
import rospy
from std_msgs.msg import String
from nao_alproxy import NaoALProxy
import sys
import time
import numpy as np


class NaoListenerNodeMultiple():
    def __init__(self, nao_ip=['192.168.0.101', '192.168.0.102']):
    # def __init__(self, nao_ip=['192.168.0.101']):
    # def __init__(self, nao_ip=[]): # test
    # def __init__(self, nao_ip):
        # input is an array of ip's (in strings)
        self.nao_alproxy = []
        self.publisher = []
        for i, ip in enumerate(nao_ip):
            self.nao_alproxy.append(NaoALProxy(ip))
            self.nao_alproxy[-1].start_nao()
            self.publisher.append(rospy.Publisher('nao_state_%d' % i, String, queue_size=10))
            rospy.Subscriber('to_nao_%d' % i, String, self.callback_to_nao, callback_args=i)

            self.blinking_on = True
            name_subscriber_blinking = 'blinking_%d' % i
            rospy.Subscriber(name_subscriber_blinking, String, self.blinking)

        rospy.init_node('nao_listener_node')  # init a listener:
        # self.blinking(i)
        # self.blinking(i-1)

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


    def parse_behavior(self, _dict):
        return json.dumps(_dict)


    def blinking(self,i):
        while True:
            if self.blinking_on == True:
                blinking_message = self.parse_behavior({'action': 'blink'})
            self.publisher[i].publish(blinking_message)
            time_now=time.time()
            time_between_blinks =np.random.exponential(2.5)
            while self.blinking_on == True and (time.time()-time_now)<time_between_blinks:  #like sleep
                pass

    def blink(self):
        self.leds.off("leds1")
        self.leds.off("leds2")
        self.leds.off("leds3")
        self.leds.on("leds3")
        self.leds.on("leds2")
        self.leds.on("leds1")

if __name__ == '__main__':
    try:
        # from command line: first input is an optional nao_ip
        if len(sys.argv) > 1:
            nao = NaoListenerNodeMultiple(sys.argv[1:])
        else:
            nao = NaoListenerNodeMultiple()
    except rospy.ROSInterruptException:
        pass