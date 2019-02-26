#!/usr/bin/env python
# install_twisted_rector must be called before importing  and using the reactor
from kivy.support import install_twisted_reactor
install_twisted_reactor()

STUDY_SITE = 'TAU'      #'TAU'      # MIT

if STUDY_SITE == 'MIT':
    from tega import Tega
elif STUDY_SITE == 'TAU':
    from nao import Nao

from twisted.internet import reactor
from twisted.internet import protocol

import rospy
from std_msgs.msg import String

import json


class EchoProtocol(protocol.Protocol):

    def dataReceived(self, data):
        #called when the client sends a message to the server
        print("dataReceived", data)
        response = self.factory.app.handle_message(data, self)

    def sendMessage(self, msg):
        #called when sending a message back to the client
        print("sendMessage",msg)
        self.transport.write(msg)


class EchoFactory(protocol.Factory):

    def __init__(self, app):
        self.app = app
        self.protocol = EchoProtocol


from kivy.app import App
from kivy.uix.label import Label


class TwistedServerApp(App):
    publishers = {}
    factory = None
    label = None
    protocols = {} # {ip1:protocol, ip2:protocol, ip3:protocol}

    def build(self):
        self.label = Label(text="server started\n")
        self.factory = EchoFactory(self)
        reactor.listenTCP(8000, self.factory)
        rospy.init_node('twisted_node')
        #rospy.Subscriber("to_twisted", String, self.transmit_msg) #not using this
        rospy.Subscriber("to_tablet", String, self.callback_to_tablet)
        # if STUDY_SITE == 'MIT':
        #     self.publishers['tega'] = Tega(self)
        # elif STUDY_SITE == 'TAU':
        #     self.publishers['nao'] = Nao(self)
        # return self.label

    def handle_message(self, msg, protocol_in):
        print("twisted_server_ros: handle_message")
        #print("protocol_in.ip:", protocol_in.ip)
        #protocol_in_vars = vars(protocol_in)
        # JUST DID THAT TO FIND WHERE THE CLIENT IP IS
        # for item in protocol_in_vars.items():
        #     print("item:", item)
        #
        # factory = protocol_in.factory
        # for itemFactory in vars(factory).items():
        #     print("itemFactory",itemFactory)
        #
        # transport = protocol_in.transport
        # for itemTransport in vars(transport).items():
        #     print("itemTransport", itemTransport)
        #
        client_ip = protocol_in.transport.client[0] #this is how to find the client ip
        print("client_ip", client_ip)
        #if client_ip not in self.protocols_ip:
        #    print("added new protocol")
        #    self.protocols_ip.append (client_ip)
        self.protocols[client_ip]=protocol_in #if the client_ip is not in the dictionary it will be added to it.


        print('self.protocol', self.protocols)
        self.label.text = "received:  %s\n" % msg
        try:
            print("msg", msg)
           # print("msg json", json.loads(str(msg)))
            msgs = []
            spl = msg.split('}{')
            print(spl)
            for k in range(0, len(spl)):
                the_msg = spl[k]
                if k > 0:
                    the_msg = '{' + the_msg
                if k < (len(spl)-1):
                    the_msg = the_msg + '}'
                json_msg = json.loads(the_msg)
                msgs.append(json_msg)
            print (msgs)
            for m in msgs:
                for topic, message in m.items():
                    topic = str(topic)
                    message["client_ip"] = client_ip  #I'm adding ip so that on callback_nao_state I can send the message to the correct client
                    message = json.dumps(message)
                    #print("message",message, "message type", type(message))
                    self.send_message(topic, message)
        except:
            print("err")
            if 'from_twisted' not in self.publishers:
                self.publishers['from_twisted'] = rospy.Publisher('from_twisted', String, queue_size=10)
            self.publishers['from_twisted'].publish(msg)
        self.label.text += "published: %s\n" % msg

        # remove when tega is connected
        # self.protocol.sendMessage(msg)
        return msg

    def send_message(self, topic, message):
        print("twisted_server_ros: send_message")
        if topic != 'tega' and topic != 'nao':
            message = str(message)
            if topic not in self.publishers:
                self.publishers[topic] = rospy.Publisher(topic, String, queue_size=10)
        self.publishers[topic].publish(message)
        print('published to ', topic, message)


    # def transmit_msg(self, data):
    #     print('twisted_server_ros: transmit_msg', data.data)
    #     self.label.text = 'transmitting ' + str(data.data)
    #     try:
    #         # if self.protocol:
    #         #    self.protocol.sendMessage(data.data)
    #         message_json = json.loads(data.data)
    #         client_ip = str(message_json["client_ip"])
    #         self.protocols[client_ip].sendMessage(str(json.dumps(message_json)))
    #     except:
    #         print("failed to transmit_msg")

    def callback_to_tablet (self, data):
        print('twisted_server_ros: transmit_msg', data.data)
        self.label.text = 'transmitting ' + str(data.data)
        try:
            #if self.protocol:
            #    self.protocol.sendMessage(data.data)
            message_json = json.loads(data.data)
            client_ip = str(message_json["client_ip"])
            self.protocols[client_ip].sendMessage(str(json.dumps(message_json)))
        except:
            print("failed to transmit_msg")
if __name__ == '__main__':
    TwistedServerApp().run()