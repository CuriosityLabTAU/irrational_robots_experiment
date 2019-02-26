import rospy
from std_msgs.msg import String
import json
import time


class Nao:

    publisher = None
    server = None
    #animations = []
    #current_animation = None



    def __init__(self, server=None):
        self.server = server
        self.publisher = rospy.Publisher('nao_commands', String, queue_size=10)
        rospy.Subscriber('nao_state', String, self.callback_nao_state)
        # self.current_animation = None
        # self.doable_animations = None
        # self.current_animation_seq = None
        #
        # self.pose_list = ['LOOKAT_CHILD', 'LOOKAT_TABLET', 'POSE_FORWARD', 'EXCITED', 'LEFTRIGHTLOOKING', 'HAPPY_UP', 'PROUD', 'SAD']
        print('Finished initializing Nao')


    def publish(self, message):
        print('nao: publish', 'message type', type(message),'message',message)
        # message = eval(message)
        # self.current_animation_seq = message[0]
        # self.current_animation = message[1]
        # print('nao: ', message[0], message[1])
        # self.doable_animations = []
        # for robot_action in self.current_animation[1:]:
        #     if robot_action.upper() == robot_action:
        #         if robot_action in self.pose_list:
        #             self.doable_animations.append(robot_action)
        #     else:
        #         self.doable_animations.append(robot_action)
        #
        # if len(self.doable_animations) == 0:
        #     self.send_finish_animation_sequence()
        # else:
        #     for robot_action in self.doable_animations:
        #         print('robot doing action:', robot_action)
        #         self.publisher.publish(robot_action)
        self.publisher.publish(message)

    def callback_nao_state (self, data):
        print('callback_nao_state: ', data.data)
        #self.send_finish_animation_sequence(data.data)


    def send_finish_animation_sequence(self,message):
        print('send_finish_animation_sequence')
        try:
            msg = {'nao': ["express", message]}
            message_json = json.loads(message)
            client_ip = str(message_json["client_ip"])
            print('trying to send a message:', msg)
            print('server', self.server)
            print('protocol', self.server.protocols[client_ip])

            print("client_ip", client_ip)
            print("protocols[client_ip]", self.server.protocols[client_ip])
            self.server.protocols[client_ip].sendMessage(str(json.dumps(msg)))
            print('sent back: ', msg)
        except:
            print('failed to send the message...')
