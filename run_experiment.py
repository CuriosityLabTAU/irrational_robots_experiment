import threading
import os

num_of_robots = 2
# naos = ['192.168.0.100', '192.168.0.102']
naos = ['192.168.0.100']

user_number = 'test_run'

def worker0():
    os.system('roscore')

def worker1():
    os.system('python nao_comm/twisted_server_ros.py')

def worker2():
    os.system('python nao_comm/nao_ros_listener_multiple.py')

def worker3(_nao):
    os.system('python nao_comm/nao_ros.py'+ ' ' + _nao[0] + ' ' + _nao[1])

def worker4(_nao):
    os.system('python nao_comm/nao_subconscious.py'+ ' ' + _nao[0] + ' ' + _nao[1])

def worker5(user_number):
    os.system('python robot_behavior.py' + ' ' + user_number)


print('''Starting the Irrational Robots Experiment
Check that you have 2 robots.
Did you updated the IPs?
Did you choose rationality?''')

t0 = threading.Thread(target=worker1)
t0.start()
threading._sleep(5)
print('========= roscore is running =========')

t1 = threading.Thread(target=worker1)
t1.start()
threading._sleep(2)
print('========= twisted_server_ros is running =========')

t2 = threading.Thread(target=worker2)
t2.start()
threading._sleep(15)
print('========= nao_ros_listener_multiple is running =========')


if len(naos) != 1:
    nao_s = [[naos[0], '1'], [naos[1], '2']]
else:
    nao_s = [[naos[0], '1']]

for n in nao_s:
    t3 = threading.Thread(target=worker3, args=(n,))
    t3.start()
    print('========= nao_ros is running =========')

    t4 = threading.Thread(target=worker4, args=(n,))
    t4.start()
    print('========= nao_subconscious is running =========')

t5 = threading.Thread(target=worker5, args = (user_number,))
t5.start()
threading._sleep(5)
print('========= experiment is running =========')