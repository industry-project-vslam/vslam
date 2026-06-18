# Installation guide

## material requirements

- a room you want to map out (minimum 2m by 2m)
- 4 crazyflie drones with flow decks on the bottom of each drone
    - two drones have an AI deck on top of them with the [correct radio addresses](#drone-addresses-and-channels)
    - two drones have a ranger deck on top of them with the [correct radio addresses](#drone-addresses-and-channels)
- 1 Crazyradio 2.0
- a laptop or PC

## Drone setup

### Crazyradio

Plug the Crazyradio 2.0 into your laptop or pc

### Drone addresses and channels

Before running the mission, every drone needs to have the correct radio address. This can be checked and changed with the Crazyflie Client.

First open the Crazyflie Client and connect one drone at a time. It is best to only turn on one drone while changing its address, so you do not accidentally change the wrong drone.

The drones should use these addresses:

```text
M stands for multiranger deck equiped drone
AI stands for AI deck equiped drone

M1  = radio://0/80/2M/E7E7E7E701
M2  = radio://0/80/2M/E7E7E7E703
AI1 = radio://0/84/2M/E7E7E7E702
AI2 = radio://0/84/2M/E7E7E7E704
```

M1 and M2 use channel `80`. AI1 and AI2 use channel `84`. All drones use `2M` datarate. The last part of the address is different for every drone, so the program knows which drone is which.

To change the address, open the Crazyflie Client, connect to the drone, and go to the configuration page where the radio address can be changed. After changing the address, write the setting to the drone and restart it. Do this for every drone until all four drones have the correct address.

A good way to avoid mistakes is to write a small label on each drone: `M1`, `M2`, `AI1`, and `AI2`.

### Front side of the drone

For this project, the side with the start/power button is used as the front side of the drone. When the guide says that a drone has to face a certain direction, this means that the start/power button side should point in that direction.

This is important for the placement. M1 and M2 have to face opposite directions, but AI1 and AI2 both have to face the same direction as M1.

### Charging and turning on the drones

Before testing, make sure all four drones are fully charged. To charge a drone, turn it off and plug it in with a USB cable. Wait until it is charged before flying. A weak battery can cause a drone to drop during the mission or reboot in the air.

When the drones are charged, place them in the correct positions while they are still turned off. After all drones are placed correctly, turn them on one by one. Wait until each drone has started properly before running the code.

### Drone placement

Before placing the drones, make sure all drones are turned off.

Use the middle line of the room as the reference line.

### Multiranger drones

Place the two Multiranger drones close to the middle line, with a total distance of about 30 cm between them.

Place M1 about 15 cm on one side of the middle line. Place M2 about 15 cm on the other side of the middle line. M1 must have the wall on its left side. M2 must have the wall on its right side. M1 and M2 should face opposite directions.

The Multiranger drones will first map the room and create the safe zone.

### AI-deck drones

Place the two AI-deck drones further away from the middle line.

Place AI1 about 75 cm in front of the middle line. Place AI2 about 75 cm behind the middle line. AI1 and AI2 must both face the same direction as M1.

The AI drones will launch after the mapping is finished. AI1 scans one half of the safe zone, and AI2 scans the other half. If an obstacle is detected in the map, the AI drone responsible for that half will fly to safe viewpoints around the obstacle.

## Dependencies

In the source folder you have the following sub-folders

- ```Fully_drone_movement/crazyflie_wall_following_pointcloud``` or ```Fully_drone_movement\crazyflie_wall_following_pointcloud``` on Windows
- ```image_stream```
- ```map_api_adaptation```

With each sub folder as current directory you must execute the following commands

to create a virtual environment

```bash
python3.12 -m venv .venv # create a virtul environment
```

```bash
source .venv/bin/activate # to enter the virtual environment... for Windows use: .venv\Scripts\activate
```

```bash
pip install -r requirements.txt # to install all required dependencies to run the apps
```

```bash
deactivate # exit the virtual environment before moving onto the next folder
```

## Running the apps

Each application must be run in a seperate terminal session.

### Map Api and Interface

from the ```map_api_adaptation``` folder

```bash
source .venv/bin/activate # enter the virtual environment... for Windows use: .venv\Scripts\activate
```

```bash
docker compose up -d --build # build the docker compose project (api and database)
```

```bash
python app_api_gradio.py # run the app on http://127.0.0.1:7860
```

### Wall following and Safezones

from the ```Fully_drone_movement/crazyflie_wall_following_pointcloud``` folder or ```Fully_drone_movement\crazyflie_wall_following_pointcloud``` folder on Windows

```bash
source .venv/bin/activate # enter the virtual environment... for Windows use: .venv\Scripts\activate
```

#### Starting the drones

With all drones are placed correctly, turn them on.

#### Running the mission

```bash
python main_v10_corners.py # run the 2D mapping mission
```

press ENTER for the mission to start

The mission will run in this order.

First, M1 and M2 connect. Then M1 and M2 take off and map the room. After the mapping is finished, a merged safe-zone map is created. If an obstacle is detected, it is exported to the safe-zone file.

After this, AI1 and AI2 connect. They both wait on the ground until they are ready. Then AI1 and AI2 launch together. AI1 scans one half of the safe zone and AI2 scans the other half. If an obstacle is present, the correct AI drone looks at it from safe viewpoints. When the AI drones are finished, they return to their own start position and land.

#### Safety controls

During the mission, the keyboard can be used for safety.

```text
L or SPACE = smooth land
E          = emergency stop
Ctrl + C   = stops the program (secondary emergency stop)
```

Use these if something goes wrong during flight.

#### Important notes

Do not move the drones after turning them on and starting the script.

Make sure the drones are fully charged before testing. A weak battery can cause a drone to drop during flight or reboot.

The `safe_zone_output` and `mission_control` folders are generated during the mission. They do not need to contain old files before starting a new run. Old files in these folders can be deleted, because new output is generated every run.

### Image stream

Connect to the AI deck wifi hotspot of your choice.

```bash
source .venv/bin/activate # enter the virtual environment... for Windows use: .venv\Scripts\activate
```

```bash
python app.py # run the app
```