from QuantLib import KalmanFilterLinear as kalman

import numpy as np
import random
import pylab as pb

# Single value example
class Voltmeter:
    def __init__(self, true_voltage, noise_level):
        self.true_voltage = true_voltage
        self.noise_level = noise_level

    def GetVoltage(self):
        return self.true_voltage

    def GetNoiseLevel(self):
        return self.noise_level

    def GetVoltageWithNoise(self):
        return random.gauss(self.GetVoltage(), self.GetNoiseLevel())

numsteps = 60

A = np.matrix([1])
H = np.matrix([1])
B = np.matrix([0])
Q = np.matrix([0.00001])
R = np.matrix([0.1])
xhat = np.matrix([3])
P = np.matrix([1])

filter = kalman.KalmanFilterLinear(A, B, H, xhat, P, Q, R)
voltmeter = Voltmeter(1.25, 0.25)

measuredvoltage = []
truevoltage = []
kalman = []

for i in range(numsteps):
    measured = voltmeter.GetVoltageWithNoise()
    measuredvoltage.append(measured)
    truevoltage.append(voltmeter.GetVoltage())
    kalman.append(filter.GetCurrentState()[0,0])
    filter.Step(np.matrix([0]), np.matrix([measured]))


pb.plot(range(numsteps), measuredvoltage, 'b', range(numsteps), truevoltage,'r', range(numsteps), kalman, 'g')
pb.xlabel('Time')
pb.ylabel('Voltage')
pb.title('Voltage Measurement with Kalman Filter')
pb.legend(('measured','true voltage','kalman'))
pb.show()