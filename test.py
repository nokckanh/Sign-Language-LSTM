import serial
import time

s = serial.Serial('com3',9600) #port is 11 (for COM12, and baud rate is 9600
time.sleep(2)    #wait for the Serial to initialize
s.write(b'Ready...')
while True:
    str = input('Enter text: ').encode()
    str = str.strip()
    if str == 'exit' :
        break
    s.write(str)