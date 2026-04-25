import serial
import time

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
time.sleep(2)

ser.write(b"A")
print("Sent: A")

time.sleep(1)

if ser.in_waiting:
    print("Received:", ser.readline().decode(errors="replace").strip())
else:
    print("No response")

ser.close()
