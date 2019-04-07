import rospy
from std_msgs.msg import String
from naoqi import ALProxy
import sys
import random
import time
import json
import threading



class NaoNode():

    def __init__(self, _robot_ip=str,_node_name=str):
        self.port = 9559
        self.robot_ip=_robot_ip
        self.node_name=_node_name

        try:
            #motionProxy
            self.motionProxy  = ALProxy("ALMotion", self.robot_ip, self.port)

            # postureProxy
            self.postureProxy = ALProxy("ALRobotPosture", self.robot_ip, self.port)

            # animatedSpeech
            self.animatedSpeech = ALProxy("ALAnimatedSpeech", self.robot_ip, self.port)

            # managerProxy
            self.managerProxy = ALProxy("ALBehaviorManager", self.robot_ip, self.port)

            #texttospeech
            self.tts = ALProxy("ALTextToSpeech", self.robot_ip, self.port)

            #trackerProxy
            self.trackerProxy = ALProxy("ALTracker", self.robot_ip, self.port)

            #AutonomousLife
            self.autonomous = ALProxy("ALAutonomousLife", self.robot_ip, self.port)

            # PostureProxy
            self.postureProxy = ALProxy("ALRobotPosture", self.robot_ip, 9559)

            # PostureProxy
            self.systemProxy = ALProxy("ALSystem", self.robot_ip, 9559)

            # PostureProxy
            self.audioplayerProxy = ALProxy("ALAudioPlayer", self.robot_ip, 9559)


            #LEDS Api:

            self.leds = ALProxy("ALLeds", self.robot_ip, self.port)
            names1 = ["FaceLed0", "FaceLed4"]
            names2 = ["FaceLed1", "FaceLed3", "FaceLed5", "FaceLed7"]
            names3 = ["FaceLed2", "FaceLed6"]
            names4 = ["ChestLeds"]


            self.leds.createGroup("leds1", names1)
            self.leds.createGroup("leds3", names3)
            self.leds.createGroup("leds2", names2)
            self.leds.createGroup("leds4", names4)


            # self.leds.on("leds4")



        except Exception,e:
            print "Could not create proxy "
            print "Error was: ",e
            sys.exit(1)

        # # un mute
        # if self.node_name != '3':
        #     self.audioplayerProxy.setMasterVolume(9)

        #autonomous_state
        self.set_autonomous_state_off()

        # wake_up
        self.wake_up()
        self.motionProxy.setBreathEnabled('Body', True)

        # if self.node_name!='3':
        #
        #     #Sitdown
        #     self.postureProxy.goToPosture("Sit", 1.0)
        #     self.leds.off("ChestLeds")

        # else:
        #     time.sleep(1.5)
        #     self.postureProxy.goToPosture("Stand", 1.0)


        #wake_up
        # self.rest()

        robot_name=self.systemProxy.robotName()

        random.seed(99)


        # #ros:
        rospy.init_node('nao_listener'+self.node_name)
        name='to_nao_subconscious_'+self.node_name
        self.publisher= rospy.Publisher('angles', String, queue_size=10)
        self.publisher_to_nao= rospy.Publisher('to_nao_'+self.node_name, String, queue_size=10)

        rospy.Subscriber(name, String, self.parse_message)
        rospy.spin()

    def sitdown(self):
        self.postureProxy.goToPosture("Sit", 0.3)

    def parse_behavior(self, _dict):
        return json.dumps(_dict)

    def parse_message(self, message):
        # message is json string in the form of:  {'action': 'run_behavior', 'parameters': ["movements/introduction_all_0",...]}
        # eval the action and run with parameters.
        # For example, eval result could look like: self,say_text_to_speech(['hello','how are you?'])
        message = str(message.data)

        # message = str(message)  #FOR TEST!!!!!!!!!!


        message_dict = json.loads(message)


        action = str(message_dict['action'])
        if 'parameters' in message_dict:
            parameters = message_dict['parameters']
        else:
            parameters = ""
        # print(str("self." + action + "('" + str(parameters) + "')"))
        eval(str("self." + action + "(" + str(parameters) + ")"))

    def run_behavior(self, parameters):
        ''' run a behavior installed on nao. parameters is a behavior. For example "movements/introduction_all_0" '''
        try:
            behavior = str(parameters[0])
            if len(parameters) > 1:
                if parameters[1] == 'wait':
                    self.managerProxy.runBehavior(behavior)
                else:
                    self.managerProxy.post.runBehavior(behavior)
            else:
                self.managerProxy.post.runBehavior(behavior)

            self.back_to_live()

        except Exception, e:
            print "Could not create proxy to ALMotion"
            print "Error was: ", e



    def say_text_to_speech (self, parameters):
        # make nao say the string text
        # parameters in the form of ['say something','say something','say something']
        for text in parameters:
            print("say_text_to_speech", text)
            self.tts.say (str(text))

    def set_autonomous_state_off(self):
        # put nao in autonomous state
        # parameters in the form of ['solitary','disabled']
        # http://doc.aldebaran.com/2-1/naoqi/core/autonomouslife.html
        self.autonomous.setState('disabled')

    def rest(self):
        self.motionProxy.rest()

    def wake_up(self):
        self.motionProxy.wakeUp()

    def open_hand(self, parameters):
        print('open_hand', parameters)
        hand_name = parameters[0]
        self.motionProxy.openHand('RHand')

    def look_at(self, parameters):
        vect2 = parameters[0]
        fractionmaxspeed = parameters[1]
        use = parameters[2]
        self.trackerProxy.lookAt(vect2, fractionmaxspeed, use)

    def point_at(self,parameters):
        print('pointAt', parameters)
        effector = 'RArm'
        vect = parameters[0]
        fractionmaxspeed = parameters[1]
        use = parameters[2]
        self.trackerProxy.pointAt(effector, vect, fractionmaxspeed, use)

    def change_pose(self, data_str):
        # data_str = 'name1, name2;target1, target2;pMaxSpeedFraction'
        # print data_str
        info = data_str.split(';')
        pNames = info[0].split(',')
        pTargetAngles = [float(x) for x in info[1].split(',')]
        # pTargetAngles = [x * almath.TO_RAD for x in pTargetAngles]  # Convert to radians
        pMaxSpeedFraction = float(info[2])
        # print(pNames, pTargetAngles, pMaxSpeedFraction)
        self.motionProxy.post.angleInterpolationWithSpeed(pNames, pTargetAngles, pMaxSpeedFraction)
        self.back_to_live()

    def change_pose_util(self, data_str):
        # data_str = 'name1, name2;target1, target2;pMaxSpeedFraction'
        # print data_str
        info = data_str.split(';')
        pNames = info[0].split(',')
        pTargetAngles = [float(x) for x in info[1].split(',')]
        # pTargetAngles = [x * almath.TO_RAD for x in pTargetAngles]  # Convert to radians
        pMaxSpeedFraction = float(info[2])
        # print(pNames, pTargetAngles, pMaxSpeedFraction)
        self.motionProxy.post.angleInterpolationWithSpeed(pNames, pTargetAngles, pMaxSpeedFraction)

    def animated_speech(self,parameters):
        # make nao say the string text
        # parameters in the form of ['say something',pitchShift=float]
        text=str(parameters)
        # pitch=parameters[1]
        print("say_text_to_animated_speech", text)
        self.animatedSpeech.say(text, {"pitchShift": 1.0})

    def get_angles(self,parameters):
        caller=self.node_name+','+parameters
        names = "Body"
        use_sensors = False
        print("get_angles")
        use_sensors = True
        # string_to_pub = json.dumps([[caller], self.motionProxy.getAngles(names, use_sensors),self.motionProxy.getBodyNames(names)])
        string_to_pub = json.dumps([[caller],self.motionProxy.getAngles(names, use_sensors)])
        print string_to_pub
        self.publisher.publish(string_to_pub)


    def move_to_pose(self,direction):
        direction=direction[0]

        if direction == 'right':
            self.change_pose_util('HeadYaw,HeadPitch,LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,LHand,LHipYawPitch,LHipRoll,LHipPitch,LKneePitch,LAnklePitch,LAnkleRoll,RHipYawPitch,RHipRoll,RHipPitch,RKneePitch,RAnklePitch,RAnkleRoll,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw,RHand;-0.88,0.01,0.93,0.26,-0.45,-1.2,0.01,0.29,-0.6,0.2,-1.53,1.41,0.84,0.0,-0.6,0.0,-1.53,1.41,0.85,0.01,0.96,-0.29,0.53,1.26,-0.05,0.3;0.2')

        elif direction == 'center':
            self.change_pose_util('HeadYaw,HeadPitch,LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,LHand,LHipYawPitch,LHipRoll,LHipPitch,LKneePitch,LAnklePitch,LAnkleRoll,RHipYawPitch,RHipRoll,RHipPitch,RKneePitch,RAnklePitch,RAnkleRoll,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw,RHand;-0.02,0.2,0.93,0.26,-0.45,-1.21,0.01,0.29,-0.6,0.2,-1.53,1.41,0.84,-0.0,-0.6,-0.2,-1.53,1.41,0.85,0.01,0.96,-0.3,0.53,1.24,-0.04,0.3;0.08')

        elif direction == 'left':
            self.change_pose_util('HeadYaw,HeadPitch,LShoulderPitch,LShoulderRoll,LElbowYaw,LElbowRoll,LWristYaw,LHand,LHipYawPitch,LHipRoll,LHipPitch,LKneePitch,LAnklePitch,LAnkleRoll,RHipYawPitch,RHipRoll,RHipPitch,RKneePitch,RAnklePitch,RAnkleRoll,RShoulderPitch,RShoulderRoll,RElbowYaw,RElbowRoll,RWristYaw,RHand;0.88,0.01,0.92,0.27,-0.47,-1.22,0.01,0.29,-0.6,0.0,-1.53,1.41,0.84,0.0,-0.6,-0.2,-1.53,1.41,0.85,0.01,0.96,-0.3,0.53,1.24,-0.04,0.3;0.2')

        self.back_to_live()

    def look_to_other_way(self,relative_to):
        relative_to=relative_to[0]


        angles=self.motionProxy.getAngles("Body", True)
        basepose_HeadYaw = angles[0]
        basepose_HeadPitch = angles[1]

        if relative_to == "right":
            self.change_pose_util('HeadYaw,HeadPitch;' + str(basepose_HeadYaw + 1.18) + ',' + str(basepose_HeadPitch - 0.2) + ';0.18')
        elif relative_to == "center":
            sign = random.choice((-1, 1))
            self.change_pose_util('HeadYaw,HeadPitch;' + str(basepose_HeadYaw + sign * (0.4)) + ',' + str(basepose_HeadPitch + 0.25) + ';0.18')
        elif relative_to == "left":
            self.change_pose_util('HeadYaw,HeadPitch;' + str(basepose_HeadYaw - 1.18) + ',' + str(basepose_HeadPitch - 0.2) + ';0.18')

        self.back_to_live()

    def agree(self):
        counter = 0
        angles=self.motionProxy.getAngles("Body", True)
        basepose = angles[1]

        for i in range(1, 5):
            self.change_pose_util('HeadPitch;' + str(basepose + 0.3 / i) + ';' + str(0.16))
            time.sleep(0.3)
            self.change_pose_util('HeadPitch;' + str(basepose - 0.15 / i) + ';' + str(0.14))
            time.sleep(0.3)
        self.change_pose_util('HeadPitch;' + str(basepose+0.05) + ';0.05')

        self.back_to_live()

    def open_hands(self):
        self.managerProxy.post.runBehavior('social_curiosity2/open_hands')
        counter = 0
        angles = self.motionProxy.getAngles("Body", True)
        basepose = angles[1]
        time.sleep(2)
        for i in range(1, 5):
            self.change_pose_util('HeadPitch;' + str(basepose + 0.3 / i/2) + ';' + str(0.16))
            time.sleep(0.3)
            self.change_pose_util('HeadPitch;' + str(basepose - 0.15 / i/2) + ';' + str(0.14))
            time.sleep(0.3)
        self.change_pose_util('HeadPitch;' + str(basepose + 0.05) + ';0.05')

        self.back_to_live()


    def look_down(self):
        angles = self.motionProxy.getAngles("Body", True)
        basepose = angles[1]
        self.change_pose_util('HeadPitch;' + str(basepose + 0.2) + ';0.08')
        time.sleep(3)
        self.change_pose_util('HeadPitch;' + str(basepose) + ';0.08')

        self.back_to_live()

    def look_up(self,side):
        angles = self.motionProxy.getAngles("Body", True)
        basepose = angles[1]

        side=str(side[0])
        b='social_curiosity2/point_'+side+'_look_up'

        self.change_pose_util('HeadPitch;' + str(basepose - 0.4) + ';0.08')

        self.managerProxy.post.runBehavior(b)




        self.back_to_live()


    def disagree(self):
        counter = 0
        angles=self.motionProxy.getAngles("Body", True)
        basepose = angles[0]
        print basepose
        for i in range(1,5):
            self.change_pose_util('HeadYaw;' + str(basepose + 0.3/i) + ';'+str(0.16-0.000*i))
            time.sleep(0.3)
            self.change_pose_util('HeadYaw;' + str(basepose - 0.3/i) + ';'+str(0.16-0.000*i))
            time.sleep(0.3)

        self.change_pose_util('HeadYaw;' + str(basepose) + ';0.1')
        self.back_to_live()


    def open_hands_neg(self):
        self.managerProxy.post.runBehavior('social_curiosity2/open_hand_neg')
        time.sleep(1.5)
        counter = 0
        angles = self.motionProxy.getAngles("Body", True)
        basepose = angles[0]
        print basepose
        for i in range(1, 5):
            self.change_pose_util('HeadYaw;' + str(basepose + 0.3 / i/2) + ';' + str(0.16 - 0.000 * i))
            time.sleep(0.3)
            self.change_pose_util('HeadYaw;' + str(basepose - 0.3 / i/2) + ';' + str(0.16 - 0.000 * i))
            time.sleep(0.3)

        self.change_pose_util('HeadYaw;' + str(basepose) + ';0.1')
        self.back_to_live()



    def hands_on_hips(self):
        self.managerProxy.post.runBehavior('social_curiosity2/hands_on_hips')
        counter = 0
        angles = self.motionProxy.getAngles("Body", True)
        basepose = angles[0]
        print basepose
        for i in range(1, 5):
            self.change_pose_util('HeadYaw;' + str(basepose + 0.3 / i) + ';' + str(0.16 - 0.000 * i))
            time.sleep(0.3)
            self.change_pose_util('HeadYaw;' + str(basepose - 0.3 / i) + ';' + str(0.16 - 0.000 * i))
            time.sleep(0.3)

        self.change_pose_util('HeadYaw;' + str(basepose) + ';0.1')
        self.back_to_live()

    def blink(self):
        self.leds.off("leds1")
        self.leds.off("leds2")
        self.leds.off("leds3")
        self.leds.on("leds3")
        self.leds.on("leds2")
        self.leds.on("leds1")


    def negative_eyes(self):
        self.leds.off("leds1")
        self.leds.off("leds2")
        self.leds.off("leds3")
        self.leds.on("leds3")
        self.leds.on("leds2")
        self.leds.on("leds1")

    def close_eyes(self):
        rDuration = 0.75
        self.leds.post.fadeRGB("FaceLed0", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed1", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed2", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed3", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed4", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed5", 0x000000, rDuration)
        self.leds.post.fadeRGB("FaceLed6", 0x000000, rDuration)
        self.leds.fadeRGB("FaceLed7", 0x000000, rDuration)
        time.sleep(2)
        self.leds.on("leds3")
        self.leds.on("leds2")
        self.leds.on("leds1")




    def move_head_naturally(self,_current_relationship):
        current_relationship=float(_current_relationship[0])
        if current_relationship== -2:
            factor=1
        else:
            if current_relationship == -1:
                factor = 1
            elif current_relationship == 0:
                factor = 0.5
            else:
                factor =0.3

        angles=self.motionProxy.getAngles("Body", True)
        basepose_HeadYaw = angles[0]
        basepose_HeadPitch = angles[1]

        self.change_pose_util('HeadYaw,HeadPitch;' + str(min(max(basepose_HeadYaw + factor*random.uniform(-0.25,0.25),-1.05),1.05)) +',' + str(min(max(basepose_HeadYaw + factor*random.uniform(-0.1,0.1),-0.15),0.3)) + ';0.045')

    def back_to_live(self):
        time.sleep(2)
        self.publisher_to_nao.publish(self.parse_behavior({'action':'natural_motion'}))

    def end_work(self):

        if self.node_name != '3':
            # Sitdown
            self.postureProxy.goToPosture("Sit", 5.0)

            # sound back
            self.audioplayerProxy.setMasterVolume(0.8)

        else:
            self.postureProxy.goToPosture("Crouch", 5.0)

        # rest
        self.rest()

# strat=NaoNode('192.168.0.100','1')
strat=NaoNode(sys.argv[1],sys.argv[2])

#
# strat=NaoNode('192.168.0.100','right')  #FOR TEST!!!!!!!!!!
# strat.look_up(['left'])                      #FOR TEST!!!!!!!!!!
# # strat.open_hands()                      #FOR TEST!!!!!!!!!!



