import numpy as np
import pandas as pd
import pickle
from quantum_calculations import *
from itertools import combinations
from random import shuffle
import json

# ### read the story from json
# stories = json.load(open('story_dict.json'))


### from qubits number to representation in the code
hq = {0:'a', 1:'b', 2:'c', 3:'d'}

### questions and the qubits they ask about
information = {'0': {'qq': [0, 1], 'qtype': 'ranking'},
               '1': {'qq': [2, 3], 'qtype': 'ranking'},
               '2': {'qq': [0, 2], 'qtype': 'ranking'},
               '3': {'qq': [1, 3], 'qtype': 'ranking'},}


### initial psi for evereyone
plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
psi0 = np.kron(plus, plus).reshape(16,1)

### dummy varibales for checking the code
stories = ['a', 'b', 'c', 'd']

### hs for testing
hs1 = {'ha': [.5], 'hb': [.4], 'hab': [.8],
       'hc': [.2], 'hd': [.1], 'hcd': [.9],
       'hac': [.5], 'had': [.5], 'hbc': [.5], 'hbd': [.5], 'hcd': [.5]}

hs2 = {'ha': [.2], 'hb': [.3], 'hab': [.9],
       'hc': [.2], 'hd': [.4], 'hcd': [.7],
       'hac': [.5], 'had': [.5], 'hbc': [.5], 'hbd': [.5], 'hcd': [.5]}

def robots_answering_order():
    robots = [1,2]
    return shuffle(robots)

def intialize_robots_H(rationality, hs = None):
    H = {'ha': [], 'hb': [], 'hab': [],
         'hc': [], 'hd': [], 'hcd': [],
         'hac': [], 'had': [], 'hbc': [], 'hbd': [], 'hcd': []}
    if hs == None:
        if rationality  == 'rational':
            H['hab'] = 1
        elif rationality  == 'irrational':
            H['hab'] = 2
    else:
        for k, v in hs.items():
            H[k] = v
    return H


def extract_info_from_buttons(person_buttons, question_type):
    if question_type == 'ratings':
        person_probs = []
        person_probs['A']   = person_buttons[0]
        person_probs['B']   = person_buttons[1]
        person_probs['A_B'] = person_buttons[2]
        return person_probs
    elif question_type == 'rankings':
        perosn_rankings = []
        for i, button in enumerate(person_buttons):
            perosn_rankings[i] = person_buttons[i]
        return perosn_rankings
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
    return ['h' + s for s in [i, j, ij]]


def update_H(H_robot, H_person, question_qubits, update = 'robot'):
    '''Update the robot's H to be equal to the person.'''
    hs = question_h_keys(question_qubits)
    for h in hs:
        if update == 'robot':
            H_robot[h] = H_person[h]
        else:
            H_person[h] = H_robot[h]
    return H_robot, hs

def generate_robot_behavior(which_robot, ps, type = 'ranking'):
    '''
    send to the robot (with ROS) what to do.
    :param ps: probabilities
    :param type: ranking/ rating
    :return:
    '''
    pass

def robot_behavior(which_robot, H_robot, H_person, question_qubits = [0,1], psi = psi0):

    H_robot, hs = update_H(H_robot, H_person, question_qubits, update = 'robot')

    full_h = [H_robot[hs[0]], H_robot[hs[1]], H_robot[hs[2]]]

    ### calculate robots probability
    ps = robot_probs(psi, full_h, question_qubits, 'conj', n_qubits=4)

    total_H = compose_H(full_h, question_qubits, n_qubits=4)
    psi_final = get_psi(total_H, psi)

    generate_robot_behavior(which_robot, ps)

    return H_robot, psi_final


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
        full_h = ['h' + H[hq[comb[0]]],
                  'h' + H[H[hq[comb[1]]]],
                  'h' + H[H[hq[comb[0]]]] + H[H[hq[comb[1]]]]]
        ps = robot_probs(full_h, comb, 'conj', psi, n_qubits=4)

        rankings[str(comb[0])] = ps[0]
        rankings[str(comb[1])] = ps[1]
        rankings[str(comb[0]) + str(comb[1]) + 'c'] = ps[2]
        # rankings[str(comb[0]) + str(comb[1]) + 'd'] = ps[3]

    df_ranks = pd.DataFrame.from_dict(rankings)
    df_ranks = df_ranks.T
    df_ranks.columns = ['prob']
    df_ranks = df_ranks.sort_values('prob')
    probs_rankings = list(df_ranks.index)

    return probs_rankings

def present_info(stories, story = None):
    '''
    send to robot/ tablet the story to present
    :return:
    '''

    ### for test run:
    print(stories[i])

def get_from_kivi():
    np.random.randint(0,1)

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

    H1 = intialize_robots_H(rationality='rational', hs = hs1) # hs - to create the ir/rationality
    H2 = intialize_robots_H(rationality='irrational', hs = hs2)

    robots = {'H': {1:H1, 2:H2},
              'state' : {1: robot1_state, 2: robot2_state}}

    ### present story 1
    present_info(stories, 0)
    cq = information['0']
    # --> ask the person to rate the probabilities
    person_buttons = get_from_kivi()

    H_person, person_state['1'] = person_parameters(person_buttons, person_state = person_state['0'],
                                                    question_qubits = cq['qq'], question_type = cq['qtype'])

    ### pesent more information
    present_info(stories, 1)
    cq = information['1']

    answering_order = robots_answering_order()

    ### generate robots answer
    for r in answering_order:
        robots['H'][r], robots['state'][r]['1'] = robot_behavior(1, robots['H'][r], H_person, question_qubits=cq['qq'], psi=robots['state'][r]['0'])

    ### Ask the person which robot do you agree with?
    person_buttons = get_from_kivi()
    chosen_robot = extract_info_from_buttons(person_buttons, question_type = 'agree')

    ### ask the person to give ratings to asked qubits
    cq = information['2']
    person_buttons = get_from_kivi()
    H_person_, person_state['2'] = person_parameters(person_buttons, person_state=person_state['0'],
                                                     question_qubits=cq['qq'], question_type=cq['qtype'])
    ### update current H of the person
    H_person = update_H(H_person, H_person_, update='person')

    ### present new information --> story 3
    present_info(stories, 3)
    cq = information['3']
    Uq = get_U_question()
    ### propogate the state using U

    ### generate robots answer
    for r in answering_order:
        robots['H'][r], robots['state'][r]['2'] = robot_behavior(1, robots['H'][r], H_person, question_qubits=cq['qq'], psi=robots['state'][r]['1'])

    ### Ask the person which robot do you agree with?
    person_buttons = get_from_kivi()
    chosen_robot = extract_info_from_buttons(person_buttons, question_type='agree')

    ### ask the person to rank all the probabilites and the conjunction between them.
    person_buttons = get_from_kivi()
    person_rankings = extract_info_from_buttons(person_buttons, question_type='ranking')

    ### robots gives ranking
    for r in answering_order:
        rankings = robots_rankings(robots['H'][r], psi=robots['state'][r]['2'])
        # todo: robot presents the rankings
        # robots['H'][r], robots['state'][r]['2'] = robot_behavior(1, robots['H'][r], H_person, question_qubits=cq['qq'], psi=robots['state'][r]['1'])


    ### The detective ask the person who did it?
    person_buttons = get_from_kivi()
    ### Based on the answer we would know if one changed the ranking after seeing the robots rankings,
