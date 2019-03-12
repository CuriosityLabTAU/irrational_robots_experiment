import subprocess

# https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python

process = subprocess.Popen(["python", "app_v1.py"], stdout=subprocess.PIPE)

# to read the latest output from the app
output = process.stdout.readline()


# while True:
#     output = process.stdout.readline()
#     if output == '' and process.poll() is not None:
#         break
#     if output:
#         print(output.strip())
#     rc = process.poll()