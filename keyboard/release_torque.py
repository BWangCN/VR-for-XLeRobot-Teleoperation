from lerobot.motors.feetech import FeetechMotorsBus, Motor, MotorCalibration

motors = {
    'm1': Motor(id=1, model='sts3215', calibration=None),
    'm2': Motor(id=2, model='sts3215', calibration=None),
    'm3': Motor(id=3, model='sts3215', calibration=None),
    'm4': Motor(id=4, model='sts3215', calibration=None),
    'm5': Motor(id=5, model='sts3215', calibration=None),
    'm6': Motor(id=6, model='sts3215', calibration=None),
}

port = input("Enter COM port (e.g. COM7): ").strip()
bus = FeetechMotorsBus(port=port, motors=motors)
bus.connect()
bus.write('Torque_Enable', 0)
print('Torque disabled — arm should be limp now.')
bus.disconnect()
