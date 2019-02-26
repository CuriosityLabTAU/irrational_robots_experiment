### general imports
import numpy as np
import pandas as pd
import pickle

### my calculations todo: need to update once the model is finished
from quantum_calculations import *

### iterations imports
from itertools import combinations
from random import shuffle
import json

### imports for the robot
import rospy
from std_msgs.msg import String

# ### read the story from json
# stories = json.load(open('story_dict.json'))

### path to sound files on the robot
robot_sound_path = '/home/nao/naoqi/sounds/HCI/'
### path to behaviors on the robot
robot_behavior_path = 'facilitator-6ea3b8/'

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

def robots_answering_order():
    order = [1,2]
    shuffle(order)
    return order


def run_robot_behavior(robots_publisher, which_robot, message):
    robots_publisher[which_robot].publish(json.dumps(message))

def callback_nao_state(data):
    if 'register tablet' not in data.data and 'sound_tracker' not in data.data:
        waiting = False
        waiting_robot = False

        try:
            signal = json.loads(data.data)['parameters'][0]
            # robot_end_signal[signal] = True
        except:
            pass

def intialize_robots_comm(nrobots):
    ### intiliaze the communication with the robots
    robots_publisher = []
    for i in range(nrobots):
        robots_publisher.append(rospy.Publisher('to_nao_%d' % i, String, queue_size=10))
        rospy.init_node('manager_node')  # init a listener:
        rospy.Subscriber('nao_state_%d' % i, String, callback_nao_state)
    return robots_publisher

def intialize_robots_H(rationality, hs = None):
    H = {'h_a': [], 'h_b': [], 'h_ab': [],
         'h_c': [], 'h_d': [], 'h_cd': [],
         'h_ac': [], 'h_ad': [], 'h_bc': [], 'h_bd': [], 'h_cd': []}
    if hs == None:
        if rationality  == 'rational':
            H['h_ab'] = 1
        elif rationality  == 'irrational':
            H['h_ab'] = 2
    else:
        for k, v in hs.items():
            H[k] = v
    return H


def extract_info_from_buttons(person_buttons, question_type):
    if question_type == 'rate':
        person_probs = {}
        person_probs['A']   = person_buttons[0]
        person_probs['B']   = person_buttons[1]
        person_probs['A_B'] = person_buttons[2]
        return person_probs
    elif question_type == 'rank':
        person_rankings = person_buttons.copy()
        for i, button in enumerate(person_buttons):
            person_rankings[i] = person_buttons[i]
        return person_rankings
    elif question_type == 'agree':
        if person_buttons[0] == 1:  # left
            return 1
        else:  # right
            return 2


def person_parameters(person_buttons = None, person_state = None, question_qubits = [0,1], question_type = 'ratings'):
    all_q = question_qubits
    person_probs = extract_info_from_buttons(person_buttons, question_type)

    ### calculate {h} from person buttons (and psi_after).

    H_person, psi_after_question = get_question_H(person_state, all_q, person_probs)
    return H_person, psi_after_question


def question_h_keys(question_qubits):
    i = hq[question_qubits[0]]
    j = hq[question_qubits[1]]
    ij = i+j
    return ['h_' + s for s in [i, j, ij]]


def update_H(H_robot, H_person, question_qubits, update = 'robot'):
    '''Update the robot's H to be equal to the person.'''
    hs = question_h_keys(question_qubits)
    for h in H_person.keys():
        if update == 'robot':
            H_robot[h] = H_person[h]
        else:
            H_person[h] = H_robot[h]
    return H_robot, hs

def generate_robot_behavior(robots_publisher, which_robot, ps, question_qubits, qtype = 'rate', test = True):
    '''
    send to the robot (with ROS) what to do.
    :param ps: probabilities
    :param type: ranking/ rating
    :return:
    '''
    if not test:
        if qtype == 'rate':
            ### turn qubits to letters.
            probs = []
            for q in question_qubits:
                probs.append(hq[q])
            probs.append(hq[0] + '_and_' + hq[1])

            ### possible intros to answers to choose from. todo: a behavior file for each
            answer_intro = ['for_my_opinion', 'i_think_that', 'my_opinion_is', 'it_seems_that']

            ### choose the intro randomly
            i = np.random.randint(0, len(answer_intro))
            action = {'action': 'run_behavior', 'parameters': ['irrational/%s' % answer_intro[i], 'wait']}
            run_robot_behavior(robots_publisher, which_robot, action)

            for i, p in enumerate(probs):
                action = {'action': 'run_behavior', 'parameters': ['irrational/suspect_%s' % p, 'wait']}
                run_robot_behavior(robots_publisher, which_robot, action)

                the_action = robot_sound_path + '%d.wav' % int(10 * np.round(ps[i]*10).flatten()[0])
                nao_message = {"action": 'play_audio_file',
                               "parameters": [the_action]}
                run_robot_behavior(robots_publisher, which_robot, nao_message)

        elif qtype == 'rank':
            ### possible intros to answers to choose from.
            answer_intro = ['my_ranking_is', 'i_think_the_ranking_is']
            ### choose the intro randomly
            i = np.random.randint(0, len(answer_intro))
            action = {'action': 'run_behavior', 'parameters': ['irrational/%s' % answer_intro[i], 'wait']}
            run_robot_behavior(robots_publisher, which_robot, action)

            for i, p in enumerate(ps):
                action = {'action': 'run_behavior', 'parameters': ['irrational/suspect_%s' % p, 'wait']}
                run_robot_behavior(robots_publisher, which_robot, action)


def robot_behavior(robots_publisher, which_robot, H_robot, H_person, question_qubits = [0,1], psi = psi0, qtype = 'rate', robots = None):

    if qtype == 'rate':
        H_robot, hs = update_H(H_robot, H_person, question_qubits, update = 'robot')

        full_h = [H_robot[hs[0]], H_robot[hs[1]], H_robot[hs[2]]]

        ### calculate robots probability
        ps = robot_probs(psi, full_h, question_qubits, 'conj', n_qubits=4)

        total_H = compose_H(full_h, question_qubits, n_qubits=4)
        psi_final = get_psi(total_H, psi)

        generate_robot_behavior(robots_publisher, which_robot, ps, question_qubits, qtype)

        return H_robot, psi_final
    elif qtype == 'rank':
        robots['rankings'][which_robot], _ = robots_rankings(robots['H'][which_robot], psi=robots['state'][which_robot]['2'])
        generate_robot_behavior(robots_publisher, which_robot, robots['rankings'][which_robot], question_qubits, qtype)
        return robots



def robot_probs(psi, full_h, all_q, fallacy, n_qubits = 4):
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

def robots_rankings(H, psi):
    combs = combinations([0, 1, 2, 3], 2)
    rankings = {}
    for comb in combs:
        comb = list(comb)
        q1, q2  = hq[comb[0]], hq[comb[1]]
        q12 = q1 + q2
        full_h = [H['h_' + q1],
                  H['h_' + q2],
                  H['h_' + q12]]
        ps = robot_probs(psi, full_h, comb, 'conj', n_qubits=4)

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

def present_info(story_dict, story = None, suspects = None, test = True):
    '''
    send to robot/ tablet the story to present
    :return:
    '''
    ### for test run:
    if story == 'rate':
        s = 'Rate the probabilities that suspects %s and %s are guilty.' % (hq[suspects[0]].capitalize(), hq[suspects[1]].capitalize())
    else:
        s = story_dict[story]
    print(s)
    # raw_input('press any key continue')


def get_from_kivi(test = True, qtype = 'rate'):
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
        pass


def get_U_question():
    np.eye(16,16)

def flow():
    ### initialize robot quantum state according to chosen rationality.
    ### H its a dictionary (array) of all hamiltonians

    person_state = {}
    robot1_state = {}
    robot2_state = {}

    person_state['0'] = psi0
    robot1_state['0'] = psi0
    robot2_state['0'] = psi0

    ### start communication with the robots
    # robots_publisher = intialize_robots_comm(2)
    robots_publisher = []

    H1 = intialize_robots_H(rationality='rational', hs = hs1) # hs - to create the ir/rationality
    H2 = intialize_robots_H(rationality='irrational', hs = hs2)

    robots = {'H': {1:H1, 2:H2},
              'state' : {1: robot1_state, 2: robot2_state},
              'rankings': {1:[], 2:[]}}

    ### present story 1
    present_info(story_dict, 0)
    cq = information['0']
    present_info(story_dict, cq['qtype'], cq['qq'])
    # --> ask the person to rate the probabilities
    person_buttons = get_from_kivi(qtype = cq['qtype'])

    H_person, person_state['1'] = person_parameters(person_buttons, person_state = person_state['0'],
                                                    question_qubits = cq['qq'], question_type = cq['qtype'])

    ### pesent more information
    present_info(story_dict, 1)
    cq = information['1']

    answering_order = robots_answering_order()

    ### generate robots answer
    for r in answering_order:
        robots['H'][r], robots['state'][r]['1'] = robot_behavior(robots_publisher,1, robots['H'][r], H_person, question_qubits=cq['qq'], psi=robots['state'][r]['0'])


    ### Ask the person which robot do you agree with?
    present_info(story_dict, 'agree')
    person_buttons = get_from_kivi(qtype ='agree')
    chosen_robot = extract_info_from_buttons(person_buttons, question_type = 'agree')

    ### ask the person to give ratings to asked qubits
    cq = information['2']
    present_info(story_dict, cq['qtype'], cq['qq'])
    person_buttons = get_from_kivi(qtype = cq['qtype'])
    H_person_, person_state['2'] = person_parameters(person_buttons, person_state=person_state['0'],
                                                     question_qubits=cq['qq'], question_type=cq['qtype'])
    ### update current H of the person
    H_person, _ = update_H(H_person, H_person_, cq['qq'], update='person')

    ### present new information --> story 3
    present_info(stories, 3)
    cq = information['3']
    Uq = get_U_question()
    ### propogate the state using U

    ### generate robots answer
    for r in answering_order:
        robots['H'][r], robots['state'][r]['2'] = robot_behavior(robots_publisher, 1, robots['H'][r], H_person, question_qubits=cq['qq'], psi=robots['state'][r]['1'])

    ### Ask the person which robot do you agree with?
    present_info(story_dict, 'agree')
    person_buttons = get_from_kivi(qtype = 'agree')
    chosen_robot = extract_info_from_buttons(person_buttons, question_type='agree')

    ### ask the person to rank all the probabilites and the conjunction between them.
    present_info(story_dict, cq['qtype'], cq['qq'])
    person_buttons = get_from_kivi(qtype = cq['qtype'])
    person_rankings = extract_info_from_buttons(person_buttons, question_type=cq['qtype'])

    ### robots gives ranking
    for r in answering_order:
        # robots['rankings'][r], _ = robots_rankings(robots['H'][r], psi=robots['state'][r]['2'])

        # todo: robot presents the rankings
        robots = robot_behavior(robots_publisher, 1, robots['H'][r],
                       H_person, question_qubits=cq['qq'], psi=robots['state'][r]['1'], qtype = cq['qtype'], robots = robots)


    ### The detective ask the person who did it?
    present_info(story_dict, 'suspect')
    person_buttons = get_from_kivi(qtype = 'suspect')
    ### Based on the answer we would know if one changed the ranking after seeing the robots rankings,

flow()