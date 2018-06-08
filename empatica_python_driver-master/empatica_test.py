# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:16:30 2017
Small program to read data from the Empatica.

@author: Ilkka
"""

import argparse
import random
import time
import sys
import socket
import signal
import datetime

from pythonosc import osc_message_builder
from pythonosc import udp_client
from struct import *




EMPATICA_ADDRESS = "127.0.0.1"
EMPATICA_PORT = 9999

#ADDRESS1 = "192.168.1.112"
ADDRESS1 = "127.0.0.1"

PORT1 = 9007

#OSCADDRESS = "/empatica"


if __name__ == "__main__":
# Attempt to ctrl-c work in windows.. not very succesful.
  def signal_handler(signal, frame):
      print( "caught a signal")
      global interrupter
      interrupter = True
     
  signal.signal(signal.SIGINT, signal_handler)
  signal.signal(signal.SIGTERM, signal_handler)

# Connect toe the Empatica BLE Server
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.connect((EMPATICA_ADDRESS, EMPATICA_PORT))
 
 
 ## On windows machine if you dont put that \r\n it does not recognize it..
 
 
 # sock.sendall("device_list\r\n".encode())
 
# Specify the correct device ID:
 
  sock.sendall("device_connect A219F2\r\n".encode())
 # sock.sendall("device_connect 033B64\r\n".encode())
  time.sleep(1)
  
  # Then subscribe to the physiological signals you wish to record:
  
#  sock.sendall("device_subscribe gsr ON\r\n".encode())
 # time.sleep(1)
 # sock.sendall("device_subscribe bvp ON\r\n".encode())
  time.sleep(1)
  sock.sendall("device_subscribe acc ON\r\n".encode())

#  Connect to Unity/Max whatever for real-time processing of the data:
  client1 = udp_client.SimpleUDPClient(ADDRESS1, PORT1)
  
# Read data from Empatica:
  interrupter = False
  while interrupter == False:
      
      data = sock.recv(1024).decode()

      data = data.replace(",", ".")
      sample_lines = data.split("\n")
      print("the full data: <{0}>".format(data))

    #  print("Samples are {0},{1}".format(samples[1], samples[2]))

# If the Empatica server sends more than one line (or more than one sample), parse each sample separately:
      if len(sample_lines) > 1:
          
          # ignore the last "line" as it is actually just the carriage return
          # splitlines above separate \n\r into two lines...
          for i in range(0, len(sample_lines) - 1):
       #       print("line numer {0} is: {1}".format(i, sample_lines[i]))
              samples = sample_lines[i].split(" ")
              
              # Maybe do some more elegant solution later but if elses it is for now
              if samples[0] == "E4_Gsr":
                  msg = osc_message_builder.OscMessageBuilder(address = "/empatica/EDA")
        #          print("sample is as float {0}".format(float(samples[1])))
            
                  daitti = datetime.datetime.fromtimestamp(float(samples[1]))
         #         print("timestamp is {0}".format(daitti))
                  teh_time = daitti.time();
      
                  msg.add_arg(teh_time.__str__());
                  msg.add_arg(samples[2])
                  msg = msg.build()
                  client1.send(msg)
                
              if samples[0] == "E4_Bvp":
                  msg = osc_message_builder.OscMessageBuilder(address = "/empatica/BVP")
          #        print("sample is as float {0}".format(float(samples[1])))
            
     
                  daitti = datetime.datetime.fromtimestamp(float(samples[1]))
           #       print("timestamp is {0}".format(daitti))
                  teh_time = daitti.time();
                     
                  msg.add_arg(teh_time.__str__());
                  msg.add_arg(samples[2])
                  msg = msg.build()
                  client1.send(msg)
                 
              if samples[0] == "E4_Ibi":
                  msg = osc_message_builder.OscMessageBuilder(address = "/empatica/IBI")
                  daitti = datetime.datetime.fromtimestamp(float(samples[1]))

                  teh_time = daitti.time();
                 # teh_time.
               #   msg.add_arg((daitti.second + (daitti.microsecond/1000000)))
                  msg.add_arg(teh_time.__str__());
                  msg.add_arg(samples[2])
                  msg = msg.build()
                  client1.send(msg)
         
              if samples[0] == "E4_Acc":
                  print("The raw data is {0}".format(samples[1]) )
                  msg = osc_message_builder.OscMessageBuilder(address = "/empatica/acc")
                  print("sample is as float {0}".format(float(samples[1])))
            
                  daitti = datetime.datetime.fromtimestamp(float(samples[1]))
                  print("timestamp is {0}".format(daitti))
                  teh_time = daitti.time();
                  print("The acc values are {0}-{1}-{2}".format(samples[2], samples[3], samples[4]))
                 # teh_time.
               #   msg.add_arg((daitti.second + (daitti.microsecond/1000000)))
#                  msg.add_arg(teh_time.__str__());
                  msg.add_arg(abs(int(samples[2])/200))
                  msg.add_arg(abs(int(samples[2])/200))
                  msg.add_arg(abs(int(samples[2])/200))
                  msg = msg.build()
                  client1.send(msg)
      #            client1.send_message("/empatica/acc", "1, 2, 3")
#         
       