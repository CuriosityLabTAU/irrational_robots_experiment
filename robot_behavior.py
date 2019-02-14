import numpy as np
import pandas as pd
import pickle

#### initialize robot quantum state according to chosen rationality.
# H its a dictionary (array) of all hamiltonians
H1 = get_H_from_rationality(rationality1)
H2 = get_H_from_rationality(rationality2)

def get_H_from_rationality(rationality):
    if rationality  == 'rational':
        H = {'ha': [], 'hb': [], 'hab': [],
             'hc': [], 'hd': [], 'hcd': [],
             'hac': [], 'had': [], 'hbc': [], 'hbd': [],  'hcd': []}
    elif rationality  == 'irrational':
        H = {'ha': [], 'hb': [], 'hab': [],
             'hc': [], 'hd': [], 'hcd': [],
             'hac': [], 'had': [], 'hbc': [], 'hbd': [],  'hcd': []}
    return H

def robot_behavior(person_buttons = None, person_state = None, question_qubits = None):
    H_person = get_person_H(person_buttons, person_state)
    H_robot = update_robot_H(H_person)

    p_i, p_j, p_ij = get_robot_probabilites(H_robot, question_qubits)

    generated_behavior = generate_robot_behavior(p_i, p_j, p_ij)

    return generated_behavior