import matplotlib.pyplot as plt
from rplidar import RPLidar
import numpy as np
from time import sleep

PORT_NAME = '/dev/ttyUSB0'
MAX_DISTANCE = 2000  # now limits plot to 2 meters
DETECTION_RANGE = 2000  # only show objects within this distance

# Initialize the LiDAR
lidar = RPLidar(PORT_NAME)

# Set up the polar plot
plt.ion()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
scan_data = [0] * 360  # one entry for each degree

try:
    print("Starting LiDAR scan... Showing objects within 2 meters")
    lidar.start_motor()
    sleep(1)

    for scan in lidar.iter_measures():
        quality, angle, distance = scan[1], scan[2], scan[3]

        if 0 < distance <= DETECTION_RANGE:
            scan_data[int(angle) % 360] = distance
        else:
            scan_data[int(angle) % 360] = 0

        if int(angle) % 360 == 0:
            ax.clear()
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_ylim(0, MAX_DISTANCE)  # <-- this zooms into the inner 2m

            angles = np.radians(np.arange(360))
            distances = np.array(scan_data)

            ax.plot(angles, distances, '.', color='green', markersize=3)
            ax.set_title("Objects within 2m of LiDAR")
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\nStopping scan...")

finally:
    print("Shutting down LiDAR...")
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    plt.ioff()
    plt.show()
