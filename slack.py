# modified from https://github.com/wkwan/modelwatch/blob/master/modelwatch.py

import requests
import schedule
import time
import sys
import glob
import os

def slacktracker(folder,channel,interval):
    f = open('slacktoken.txt', 'r')
    token = f.readline()
    f.close()
    URL = "https://slack.com/api/files.upload"
    def get_latest_file_and_send_to_slack():
        try: 
            list_of_files = glob.glob(folder + "/*")
            latest_file = max(list_of_files, key=os.path.getctime)
            print("Upload the file:", latest_file)
            data = {
                "channels": channel
            }
            files = {"file": open(latest_file, 'rb')}
            headers = {
			"Authorization": "Bearer " + token
		}
    
            requests.post(URL, data=data, files=files, headers=headers)  
        except: 
            print("No file to upload")


    schedule.every(interval).minutes.do(get_latest_file_and_send_to_slack) 

    while True: 
        schedule.run_pending() 
        time.sleep(1) 

slacktracker('logs','training',1)