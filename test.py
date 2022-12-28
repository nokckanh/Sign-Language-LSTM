import serial
import time

s = serial.Serial('com3',9600) #port is 11 (for COM12, and baud rate is 9600
time.sleep(2)    #wait for the Serial to initialize
s.write(b'HIEU...')

def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1

while True:
    str = input('Enter text: ').encode()
    str = str.strip()
    if str == 'exit' :
        break
    #arr = ['a','b','c']
    #s.write(listToString(arr).encode())

    s.write(str)