# Meeting notes

## Week 01

**check client kickoff presentation**

## Week 02

```text
```

## Week 03

```text
mapping works perfectly, api image upload
stitching together drone maps, to create a common map
- create map using multiple drones
each drone sends images using wifi
each drone is granted a static ip
drone collision avoidance, using P2P radio, with set positions
sloped surface is viable, texture is important (immersive room or swimming pool)
working now to intergrate different sensors
testing using simple drone with square sequence run 10% error with accumulation
- P2P collision avoidance with > 3 drones
track progress with incremental steps, focus on documentation (remark from Bart)
nikita encountered accumulation between the drones 0.5 meters starting distance between drones is not enough
- combining ai deck map navigation position inference, and flow deck position estimate
30 seconds per image processing for map navigation
- (bart) detect drones and know which drone is being detected, know your nearest neighbour
- use homeing beacons (an object to recognize) -> redundant because of mapping positioning is underway
obstacle avoidance, problem with testing environment, not enough walls, creating cardboard environment or pillows, (bart) go slowly to avoid collision
- (bart) what would be needed for on board processing to detect obstacles (try raspberry pi)
right now using drone net for detecting obstacles, it is an optimized model but for older hardware
(for roel) look into number of parameters and compression techniques
(personal idea) using segmentation, (tofa idea) using depth models, on the drone ai deck itself
pulp tiny v3 ... but it seems the most logical choice
- look into whole application architecture also including pre run necessities and communicating systems, delays during runs
we received a multiranger deck, use it as a single example
we can detect people using yolo26n on the server, nazar improved with retraining yolo26n on the server
lighting is a big edge case for the camera
next detection step, don't forget about previously detected items, abandon navigational item detection
```

## Week 04

```text
last time the model recognized people
state classification is already implemented
using yolo26n retrained and mobilenet retrained running on the central computer

mapping different frames from different cameras is still difficult but works
in documentation all solutions have a camera that can detect depth
we will try models where the drone position is part of the input parameters
problem with combining drone relative position is depth estimation is not alligned

drone wall following with ranger decks and mapping safe zones, objects of interest and ai decks afterward

human operated swarm, setting waypoints for the drone

trying master slave drone communication
central compute system is requireed because of lack of computation
1 in front, 2 in the back

master slave relation is difficult because of undetected objects that the following ai decks can follow

relative positioning using flow deck only is not used
using two radio receives you can get positioning using the angle of reception

all teams struggled with position estimation, limitted compution, difficulty of firmware programming

a virtual environment can remove a lot of limitations our drones experience

- visual demonstration of the master slave approach
- integrate, ranger deck - ai deck approach, object detection of multiple drones, send image data to pointcloud server
    - annotationg region of interest with ai deck detection seems a bit difficult
    - focus more on random exploring
    - explain pros and cons
```

## Week 05

```text
focuss on deliverables and documentation
we will focuss on deliverables this week unfinished reasearch (ranger deck, ai decks group / improving the person state estimation model)
film systems (ranger deck first, ai decks second / image streaming, object detection, map making)
also mention failed approaches
```