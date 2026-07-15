import json

calib_path = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0001_crossing2_00/calibration.json'

with open(calib_path, 'r') as f:
    calib = json.load(f)

cam_name = 'crossing2_13_thermal_camera'
cam_data = calib[cam_name]

print(f"\n--- Deep Dive: {cam_name} ---")

print("\n[Intrinsics Keys]:")
print(cam_data['intrinsics'].keys())

print("\n[Extrinsics Keys]:")
print(cam_data['extrinsics'].keys())

# Let's see a sample of the data format
print("\n[Sample Rotation Matrix (R)]:")
# Adjust the key below based on what 'Extrinsics Keys' prints!
# I'm guessing it might be 'rotation' or 'R'import json

calib_path = '/media/pink-rabbit/Data/SmartTrafficMonitoring/experiment_v2x/data/UrbanIng-V2X/dataset/20241126_0001_crossing2_00/calibration.json'

with open(calib_path, 'r') as f:
    calib = json.load(f)

# Let's look at one of the thermal infrastructure cameras
cam_name = 'crossing2_13_thermal_camera'

if cam_name in calib:
    print(f"\n--- Calibration keys for {cam_name} ---")
    for key, value in calib[cam_name].items():
        if isinstance(value, list):
            print(f"'{key}': Matrix/List of size {len(value)}")
        else:
            print(f"'{key}': {type(value).__name__}")
else:
    print(f"Could not find {cam_name} in calibration keys. Found these instead: {list(calib.keys())[:5]}...")
