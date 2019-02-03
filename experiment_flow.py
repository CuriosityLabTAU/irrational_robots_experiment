import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class participant():
    def __init__(self, type):
        self.type = type # person ot robot
        self.probs = {'pa':[], 'pb':[], 'pc':[], 'pd':[],
                      'pab':[], 'pcb':[]}

participant = {'type' : [],
               'probs' : {'pa':[], 'pb':[], 'pc':[], 'pd':[],'pab':[], 'pcb':[]}}

#### initialize robot quantum state according to chosen rationality.
psi_robot1, H1 = get_quantum_state(rationality1)
psi_robot2, H2 = get_quantum_state(rationality2)

#### Loop that run all the stories for a specific scenario.
for story in stories():

    #### A story is presented to the person.
    present_story(story1)

    #### The person needs to rank/ give probability to the 3 options.
    p = get_answer(person) # p = [pa, pb, pab]
    update_probs(person, p, qubits, fallacy)

    #### The robots (and the person) hear a 2nd story
    present_story(story2)

    #### The two robots give their answers.
    pc1 , pd1, pcd1 = get_answer(robot1, qubits, fallacy, U_question)
    pc2 , pd2, pcd2 = get_answer(robot2, qubits, fallacy, U_question)

    #### Person choose which robot answers it agree with
    chosen_robot = choose_robot(robot1, robot2)

    #### update the person's probabilities to be equal to the robot.
    pc, pd, pcd = update_probs(person, chosen_robot, qubits)

    #### update the robots next probabilities.
    #### if the person chose its answers, todo it will take his/hers or generate according to quantum state? (U_person)
    #### if the person DIDN'T choose its answers, todo it will update the state to a new state with same rationality?

    #### optional - ask the person again on the first probabilities, to see if the robots affected it.