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

interim

```text
A drone can scan anythin
A swarm explores faster, there is a need for adaptive behaviour
current solution, master slave, ranger + follower ai deck
let ranger deck drone swarm around obstacles, create a map of environment with a large padding around obstacles to protect ai deck drones
space must be unknown
don't use deterministic models
we have access to multiple ranger decks

- explore 2 approaches
    - drone groups consisting of ranger deck and ai decks
    - ranger deck swarm first, ai deck swarm second

first only few obstacles
look into obstacle wall follower

future research, project ai deck images on ranger deck map

deliverables
    - opportunities
    - explored options
    - what detection models are runnable on ai deck
    - central ranger deck mapping
    - extra decentralized ranger deck mapping
```