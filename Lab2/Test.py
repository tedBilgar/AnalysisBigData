import time
import random
import requests
for x in range(20):
    time.sleep(0.125)
    docker_host = random.choice(['3.238.72.103', '3.235.198.44', '3.231.228.99'])
    get_request = requests.get('http://' + str(docker_host))
    print("Host: {0}, Container: {1}".format(docker_host, get_request))