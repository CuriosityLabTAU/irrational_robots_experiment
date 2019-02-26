import threading
import os

def worker1():
    os.system('python twisted_server_ros.py')

def worker2():
    os.system('python nao_ros_listener_multiple.py')

def worker3():
    os.system('python communicate_with_nao_test.py')

print('''Starting the Irrational Robots Experiment
Check that you have 2 robots.
Did you updated the IPs?
Did you choose rationality?''')

t1 = threading.Thread(target=worker1)
t1.start()
threading._sleep(2)
print('========= twisted_server_ros is running =========')

t2 = threading.Thread(target=worker2)
t2.start()
threading._sleep(15)
print('========= nao_ros_listener_multiple is running =========')

t2 = threading.Thread(target=worker3)
t2.start()
threading._sleep(5)
print('========= experiment is running =========')