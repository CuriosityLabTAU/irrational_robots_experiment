import threading
import os

num_of_robots = 2
###     first ip - red , second ip - blue
naos = ['192.168.0.100', '192.168.0.101']

# naos = ['192.168.0.101']
# naos = [[],[]]


def worker0():
    os.system('roscore')

def worker1():
    os.system('python nao_comm/twisted_server_ros.py')

def worker2(_nao):
    os.system('python nao_comm/nao_ros_listener_multiple.py'+ ' ' + _nao[0] + ' ' + _nao[1])

def worker3(_nao):
    os.system('python nao_comm/nao_ros.py'+ ' ' + _nao[0] + ' ' + _nao[1])

def worker4(_nao):
    os.system('python nao_comm/nao_subconscious.py'+ ' ' + _nao[0] + ' ' + _nao[1])

def worker5(robots_colors):
    os.system('python robot_behavior.py')
    # os.system('python robot_behavior.py' + ' ' + str(robot_colors).replace(' ', '_'))


print('''Starting the Irrational Robots Experiment
Check that you have 2 robots.
Did you updated the IPs?

Change global Gender in app_v1''')

t = threading.Thread(target=worker0)
t.start()
threading._sleep(5)
print('========= roscore is running =========')

t0 = threading.Thread(target=worker1)
t0.start()
threading._sleep(5)
print('========= roscore is running =========')

t1 = threading.Thread(target=worker1)
t1.start()
threading._sleep(2)
print('========= twisted_server_ros is running =========')


if len(naos) != 1:
    nao_s = [[naos[0], '1'], [naos[1], '2']]
else:
    nao_s = [[naos[0], '1']]

for n in nao_s:
    t3 = threading.Thread(target=worker3, args=(n,))
    t3.start()
    print('========= nao_ros is running =========')
#
    # t4 = threading.Thread(target=worker4, args=(n,))
    # t4.start()
    # print('========= nao_subconscious is running =========')

threading._sleep(5)

t2 = threading.Thread(target=worker2, args=(naos,))
t2.start()
threading._sleep(15)
print('========= nao_ros_listener_multiple is running =========')

# t5 = threading.Thread(target=worker5, args = (str(robot_colors),))
t5 = threading.Thread(target=worker5, args = (None,))
t5.start()
threading._sleep(5)
print('========= experiment is running =========')