### general imports
import numpy as np
import pandas as pd
import json
import pickle
import subprocess
import ast # str2dict

### my calculations todo: need to update once the model is finished
from quantum_calculations import *

### iterations imports
from itertools import combinations
from random import shuffle
import operator

import json

### imports for the robot
import rospy
from std_msgs.msg import String

from time import sleep
from datetime import datetime
import sys

# ### read the story from json
# stories = json.load(open('story_dict.json'))


### num of robots connected
num_robots = 1

### path to sound files on the robot
robot_sound_path = '/home/nao/naoqi/sounds/HCI/'
### path to behaviors on the robot
# robot_behavior_path = 'facilitator-6ea3b8/'
robot_behavior_path = 'torr_test_v1-3714cd/'


### global robot flow parameters
robot_end_signal = {}

### from qubits number to representation in the code
hq = {0:'a', 1:'b', 2:'c', 3:'d'}

### questions and the qubits they ask about
information = {'0': {'qq': [0, 1], 'qtype': 'rate'},
               '1': {'qq': [2, 3], 'qtype': 'rate'},
               '2': {'qq': [0, 2], 'qtype': 'rate'},
               '3': {'qq': [1, 3], 'qtype': 'rank'},}


### initial psi for evereyone
plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
psi0 = np.kron(plus, plus).reshape(16,1)

### dummy varibales for checking the code
stories = ['a', 'b', 'c', 'd'] # add questions: which robot do you agree with?, rank in descending order. etc.

story_dict = {
    0: 'A diamond shop was robbed. The police came straight away and caught a couple of people (separately) near the scene, but no diamonds were found. The detective assigned to the case is sure, with no doubt, that at least one of the suspects did it. You and the robots need to help the detective figure out who robbed the shop. here are suspects are: Suspect A: Is tall, with a black Louis Vuitton leather jacket and a Rolex watch. Suspect B: Has blonde hair and had a cut above the left eyebrow.',
    1: 'Suspect C: Tried to run from the scene when the police asked to stop. Suspect D: Is not willing to talk before a lawyer is present.',
    2: 'Suspect B told the police that she got the cut when from a tree branch when she took her dog for a walk a few hours earlier to the incident. Suspect D is still refusing to talk even when the lawyer arrived.',
    'rate': 'Rate the probabilities that each of the suspects is guilty.',
    'rank': 'Rank the suspects in a descending order according to how likely they are guilty',
    'agree': 'Which robot do you agree with',
    'suspect' : 'Who did it?'
        }

### hs for testing
hs1 = {'h_a': [.5], 'h_b': [.4], 'h_ab': [.8],
       'h_c': [.2], 'h_d': [.1], 'h_cd': [.9],
       'h_ac': [.5], 'h_ad': [.5], 'h_bc': [.5], 'h_bd': [.5], 'h_cd': [.5]}

hs2 = {'h_a': [.2], 'h_b': [.3], 'h_ab': [.9],
       'h_c': [.2], 'h_d': [.4], 'h_cd': [.7],
       'h_ac': [.5], 'h_ad': [.5], 'h_bc': [.5], 'h_bd': [.5], 'h_cd': [.5]}

class robot():
    def __init__(self, _robot_ip=str, _node_name=str, rationality = 'rational', hs = hs1):
        self.port = 9559
        self.robot_ip=_robot_ip
        self.node_name=_node_name

        self.publisher = rospy.Publisher('to_nao_%d' % _node_name, String, queue_size=10)
        rospy.init_node('manager_node')  # init a listener:
        rospy.Subscriber('nao_state_%d' % _node_name, String, self.callback_nao_state)

        self.robot1_state = {}
        self.robot1_state['0'] = psi0
        self.H = self.intialize_robots_H(rationality=rationality, hs = hs)

        action = {"action": "wake_up"}
        self.run_robot_behavior(action)

    def intialize_robots_H(self, rationality, hs=None):
        H = {'h_a': [], 'h_b': [], 'h_ab': [],
             'h_c': [], 'h_d': [], 'h_cd': [],
             'h_ac': [], 'h_ad': [], 'h_bc': [], 'h_bd': [], 'h_cd': []}
        if hs == None:
            if rationality == 'rational':
                H['h_ab'] = 1
            elif rationality == 'irrational':
                H['h_ab'] = 2
        else:
            for k, v in hs.items():
                H[k] = v
        return H


    def run_robot_behavior(self, message):
        if 'parameters' in message:
            self.signal = message['parameters'][0]
            self.robot_end_signal[self.signal] = False
        self.publisher.publish(json.dumps(message))
        if 'parameters' in message:
            while not self.robot_end_signal[self.signal]:
                pass


    def callback_nao_state(self, data):
        if 'register tablet' not in data.data and 'sound_tracker' not in data.data:
            self.waiting = False
            self.waiting_robot = False

            try:
                self.signal = json.loads(data.data)['parameters'][0]
                self.robot_end_signal[self.signal] = True
            except:
                pass

    def question_h_keys(self, question_qubits):
        i = hq[question_qubits[0]]
        j = hq[question_qubits[1]]
        ij = i + j
        return ['h_' + s for s in [i, j, ij]]

    def update_H(self, H_robot, H_person, question_qubits, update = 'robot'):
        '''Update the robot's H to be equal to the person.'''
        hs = self.question_h_keys(question_qubits)
        for h in H_person.keys():
            if update == 'robot':
                H_robot[h] = H_person[h]
            else:
                H_person[h] = H_robot[h]
        return H_robot, hs

    def generate_robot_behavior(self, ps, question_qubits, qtype = 'rate', test = True):
        '''
        send to the robot (with ROS) what to do.
        :param ps: probabilities
        :param type: ranking/ rating
        :return:
        '''

        if not test:
            if qtype == 'rate':
                ### turn qubits to letters.
                probs = {}
                for i, q in enumerate(question_qubits):
                    probs[hq[q]] = int(10 * np.round(ps[i]*10).flatten()[0])
                probs[hq[question_qubits[0]] + '_and_' + hq[question_qubits[1]]] = int(10 * np.round(ps[-1]*10).flatten()[0])

                ### arange the probs by their value.
                probs = sorted(probs.items(), key=operator.itemgetter(0))

                ### possible intros to answers to choose from. todo: a behavior file for each
                answer_intro = ['for_my_opinion', 'i_think_that', 'my_opinion_is', 'it_seems_that']

                ### choose the intro randomly
                i = np.random.randint(0, len(answer_intro))
                action = {'action': 'run_behavior', 'parameters': [robot_behavior_path + '%s' % answer_intro[i], 'wait']}
                self.run_robot_behavior(self.publisher, action)

                # log_entery(**{'state':'robot%s' % which_robot,
                #                                      'val':{'intro' : answer_intro[i],'probs' : probs}})

                for p,val in probs:
                    action = {'action': 'run_behavior', 'parameters': [robot_behavior_path + 'suspect_%s' % p, 'wait']}
                    self.run_robot_behavior(action)

                    action = {'action': 'run_behavior', 'parameters': [robot_behavior_path + 'prob%d' % val, 'wait']}
                    self.run_robot_behavior(action)

            elif qtype == 'rank':
                ### possible intros to answers to choose from.
                answer_intro = ['my_ranking_is', 'i_think_the_ranking_is']
                ### choose the intro randomly
                i = np.random.randint(0, len(answer_intro))
                action = {'action': 'run_behavior', 'parameters': [robot_behavior_path + '%s' % answer_intro[i], 'wait']}
                self.run_robot_behavior(action)

                # log_entery(**{'state':'robot%s' % which_robot,
                #                                      'val': {'intro' :answer_intro[i],'rankings' :ps}})

                for i, p in enumerate(ps[:3]):
                    action = {'action': 'run_behavior', 'parameters': [robot_behavior_path + 'suspect_%s' % p, 'wait']}
                    self.run_robot_behavior(action)


    def robot_behavior(self, H_person, question_qubits = [0,1], psi = psi0, qtype = 'rate', robots = None):

        if qtype == 'rate':
            self.H, hs = self.update_H(self.H, H_person, question_qubits, update = 'robot')

            full_h = [self.H[hs[0]], self.H[hs[1]], self.H[hs[2]]]

            ### calculate robots probability
            ps = self.robot_probs(psi, full_h, question_qubits, 'conj', n_qubits=4)

            total_H = compose_H(full_h, question_qubits, n_qubits=4)
            psi_final = get_psi(total_H, psi)

            self.generate_robot_behavior(ps, question_qubits, qtype, test = False)
            # self.state[i] = psi_final

            return psi_final
        elif qtype == 'rank':
            self.rankings, _ = self.robots_rankings(self.H, psi=self.state['2'])
            self.generate_robot_behavior(self.rankings, question_qubits, qtype, test = False)


    def robot_probs(self, psi, full_h, all_q, fallacy, n_qubits = 4):
        ### calculate the probabilities of the robot
        p_i = get_general_p(full_h, all_q, '0', psi, n_qubits=4)
        p_j = get_general_p(full_h, all_q, '1', psi, n_qubits=4)

        if fallacy == 'conj':
            p_ij = get_general_p(full_h, all_q, 'C', psi, n_qubits=4)
        elif fallacy == 'disj':
            p_ij = get_general_p(full_h, all_q, 'D', psi, n_qubits=4)
        ps = [p_i, p_j, p_ij]

        if fallacy == 'both':
            p_ijc = get_general_p(full_h, all_q, 'C', psi, n_qubits=4)
            p_ijd = get_general_p(full_h, all_q, 'D', psi, n_qubits=4)
            ps = [p_i, p_j, p_ijc, p_ijd]

        return ps


    def robots_rankings(self, psi):
        combs = combinations([0, 1, 2, 3], 2)
        rankings = {}
        for comb in combs:
            comb = list(comb)
            q1, q2  = hq[comb[0]], hq[comb[1]]
            q12 = q1 + q2
            full_h = [self.H['h_' + q1],
                      self.H['h_' + q2],
                      self.H['h_' + q12]]
            ps = self.robot_probs(psi, full_h, comb, 'conj', n_qubits=4)

            rankings[q1] = ps[0].flatten()
            rankings[q2] = ps[1].flatten()
            rankings[q1 + '_and_' + q2] = ps[2].flatten()
            # rankings[str(comb[0]) + str(comb[1]) + 'd'] = ps[3]

        df_ranks = pd.DataFrame.from_dict(rankings)
        df_ranks = df_ranks.T
        df_ranks.columns = ['prob']
        df_ranks = df_ranks.sort_index(by = 'prob')
        probs_rankings = list(df_ranks.index)

        return probs_rankings, df_ranks

def get_U_question():
    return np.eye(16,16)


def get_from_kivi(app_thread = None, test = True, qtype = 'rate'):
    if test:
        if qtype == 'agree':
            return [np.random.randint(0,2)]
        elif qtype == 'rate':
            return [np.random.randint(1,10),np.random.randint(1,10),np.random.randint(1,10)]
        elif qtype == 'rank':
            temp = np.arange(1,12)
            shuffle(temp)
            return temp
        elif qtype == 'suspect':
            return 'suspect' + hq[np.random.randint(0,4)] + 'did it!'
    else:
        ### get the real data from kivi
        output = app_thread.stdout.readline()
        return output

def extract_info_from_buttons(person_buttons, question_type, question_qubits = [0,1]):
    if question_type == 'rate':
        person_probs = {}
        person_probs['A']   = person_buttons[hq[question_qubits[0]]]
        person_probs['B']   = person_buttons[hq[question_qubits[1]]]
        person_probs['A_B'] = person_buttons[hq[question_qubits[0]] + '_and_' + hq[question_qubits[1]]]
        return person_probs
    elif question_type == 'rank':
        # person_rankings = person_buttons.copy()
        # for i, button in enumerate(person_buttons):
        #     person_rankings[i] = person_buttons[i]
        return person_buttons
    elif question_type == 'agree':
        return person_buttons
    elif question_type == 'end_app':
        return question_type

def person_parameters(person_buttons = None, person_state = None, question_qubits = [0,1], question_type = 'ratings'):
    all_q = question_qubits
    person_probs = extract_info_from_buttons(person_buttons, question_type, question_qubits)

    ### calculate {h} from person buttons (and psi_after).

    H_person, psi_after_question = get_question_H(person_state, all_q, person_probs)
    return H_person, psi_after_question

def robots_answering_order():
    order = [x+1 for x in range(num_robots)]
    shuffle(order)
    return order


def flow(user):
    ### initialize robot quantum state according to chosen rationality.
    ### H its a dictionary (array) of all hamiltonians
    global my_logger

    # log_entery(**{'state': 'event','val': 'experiment start'})

    robot1  = robot(_robot_ip='192.168.0.100', _node_name='1', rationality = 'rational', hs = hs1)
    # robot2  = robot(_robot_ip='192.168.0.103', _node_name='2', rationality = 'irrational', hs = hs2)
    robots = {1:robot1}
    # robots = {1:robot1, 2:robot2}

    person_state = {}
    person_state['0'] = psi0

    person = {'H':[],
              'state':person_state}

    # log_entery(**{'state':'event','val':'robot initialized'})

    ### INITIALIZE APP ###
    app_thread = subprocess.Popen(["python", "desktop_app/app_v1.py"], stdout=subprocess.PIPE)

    # log_entery(**{'state':'event','val':'experiment start'})

    ### present story 1
    q = '0'
    cq = information[q]
    # log_entery(**{'state':'story'+q,'val':cq})

    # --> ask the person to rate the probabilities
    person_buttons = get_from_kivi(app_thread, test = False, qtype = cq['qtype'])
    person_buttons = ast.literal_eval(person_buttons)
    # log_entery(**{'state':'person_input' + q, 'val': person_buttons})

    person['H'], person['state']['1'] = person_parameters(person_buttons, person_state = person['state']['0'],
                                                    question_qubits = cq['qq'], question_type = cq['qtype'])
    # log_entery(**{'state':'person_state' + q, 'val':person})

    ### pesent more information
    q = '1'
    cq = information[q]
    # log_entery(**{'state':'story'+q, 'val':cq})

    answering_order = robots_answering_order()
    print(answering_order)
    ### generate robots answer

    sleep(3)

    for r in answering_order:
        robots[r].H, robots[r].state['1'] = robots[r].robot_behavior(person['H'], question_qubits=cq['qq'], psi=robots[r].state['0'])

    # log_entery(**{'state':'robots_states' + q, 'val':robots})

    ### Ask the person which robot do you agree with?
    person_buttons = get_from_kivi(app_thread, test = False, qtype = cq['qtype'])
    chosen_robot = extract_info_from_buttons(person_buttons, question_type = 'agree')
    # log_entery(**{'state':'person_agree' + q, 'val':chosen_robot})

    ### ask the person to give ratings to asked qubits
    q = '2'
    cq = information[q]
    # log_entery(**{'state':'story'+q, 'val':cq})

    person_buttons = get_from_kivi(app_thread, test = False, qtype = cq['qtype'])
    person_buttons = ast.literal_eval(person_buttons)
    # log_entery(**{'state':'person_input'+q,'val':person_buttons})

    H_person_, person['state']['2'] = person_parameters(person_buttons, person_state=person['state']['0'],
                                                     question_qubits=cq['qq'], question_type=cq['qtype'])
    ### update current H of the person
    person['H'], _ = update_H(person['H'], H_person_, cq['qq'], update='person')

    # log_entery(**{'state':'person_state' + q, 'val': person})

    ### present new information --> story 3
    q = '3'
    cq = information[q]
    # log_entery(**{'state':'story'+q, 'val':cq})

    ### propogate the state using U
    # start with I - identity.
    Uq = get_U_question()
    # log_entery(**{'state':'U_matrix'+q, 'val':Uq})

    sleep(3)
    answering_order = robots_answering_order()
    ### generate robots answer
    for r in answering_order:
        robots[r].state['1'] = np.dot(Uq, robots[r].state['1'])
        robots[r].H, robots[r].state['2'] = robots[r].robot_behavior(person['H'], question_qubits=cq['qq'], psi=robots[r].state['1'])
    # log_entery(**{'state':'robots_states' + q, 'val': robots})


    ### Ask the person which robot do you agree with?
    person_buttons = get_from_kivi(app_thread, test=False, qtype=cq['qtype'])
    chosen_robot = extract_info_from_buttons(person_buttons, question_type='agree')
    # log_entery(**{'state':'person_agree' + q, 'val': chosen_robot})

    ### ask the person to rank all the probabilites and the conjunction between them.
    person_buttons = get_from_kivi(app_thread, test=False, qtype=cq['qtype'])
    person_rankings = extract_info_from_buttons(person_buttons, question_type=cq['qtype'])
    # log_entery(**{'state':'person_rankings' + q, 'val': person_rankings})

    answering_order = robots_answering_order()
    ### robots gives ranking
    for r in answering_order:
        robots[r].robot_behavior(person['H'], question_qubits=cq['qq'], psi=robots['state'][r]['1'], qtype = cq['qtype'], robots = robots)
    # log_entery(**{'state':'robots_states' + q + '_ranks', 'val': robots})

    ### The detective ask the person who did it?
    person_buttons = get_from_kivi(app_thread, test=False, qtype=cq['qtype'])
    who_did_it = extract_info_from_buttons(person_buttons, question_type=cq['qtype'])    ### Based on the answer we would know if one changed the ranking after seeing the robots rankings,
    # log_entery(**{'state':'who_did_it', 'val': who_did_it})

    ### which robot is the detective
    person_buttons = get_from_kivi(app_thread, test=False, qtype='agree')
    robot_detective = extract_info_from_buttons(person_buttons, question_type='agree')
    # log_entery(**{'state':'detective_robot', 'val': robot_detective})

    #### app closed
    person_buttons = get_from_kivi(app_thread, test=False, qtype=cq['qtype'])
    app_closed = extract_info_from_buttons(person_buttons, question_type='end_app')
    # log_entery(**{'state':'event', 'val':'experiment ended'})


    my_logger = my_logger[['time','state', 'val']]
    my_logger.to_csv('my_logger_{date:%Y-%m-%d %H:%M:%S}.csv'.format( date=datetime.datetime.now()))

    for r in range(num_robots):
        action = {"action": "rest"}
        robots[r].run_robot_behavior(action)

flow('test')
# flow(sys.argv[1])

# pickle.load(open(fname2read, 'rb'))