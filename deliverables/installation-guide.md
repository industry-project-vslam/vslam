# Installation guide

## material requirements

- a room you want to map out (minimum 2m by 2m)
- 4 crazyflie drones with flow decks on the bottom of each drone
    - two drones have an AI deck on top of them with the radio addresses: ```Noah please write the ai deck drones addresses here```
    - two drones have a ranger deck on top of them with the radio addresses: ```Noah please write the ranger deck drones addresses here```
- 1 crazyradio
- a laptop or PC

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

