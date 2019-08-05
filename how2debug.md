# How to do fast debughing.
Because I'm using multi-threading it's hard to do proper debug.\
Follow the instruction bellow,

1. app_v1.py --> ()line 8) uncomment or add this line 'mixer.init(frequency=100, size=-16, channels=2, buffer=2048)'
2. nao_listner_mode.py --> change nao_ip = [] 
3. robot_behavior.py
    1. Cancel all sleep.
    2. pass the function run_robot_behavior.

4. Run 'roscore' command in terminal.     
5. Run twisted_server_ros.py
6. Run nao_ros_listner_multiple.py
7. Run (not debug) robot_behavior.py --> print what you want to debug.

# When you are done DON'T FORGET to undo allthe changes.

#Regular RUN
Just run: run_experiment.py