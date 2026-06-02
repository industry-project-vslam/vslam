# Meeting reports

## Kickoff presentation

Check the kickoff presentation for context

### Todo

- research all communication interfaces available for the drone
- research navigational algorithms
- research available decks for the crazyflie
- research swarm navigation algorithms inspired by nature (bee colony, ant pheromones)
- research drone 2 drone communication
- research drone rotation to scan the environment

### Client demands

- the drones need spatial awareness using visual input from the ai deck camera
- the drones need to detect people and navigational elements (door, window, stairs)
- test the possibilities with the crazyflies
    - how much swarm intelligence is possible, fall back on a central compute and radio communication
- redundancy is important, the system should work for any number of drones

## Week 01

### Progress

- research all communication interfaces available for the drone
    - p2p radio
    - drone to central compute radio
    - ai deck to central compute image streaming
- research available decks for the crazyflie
    - positioning nodes/decks are not viable due to power and complexity issues
    - flow deck and ai deck are a viable combination for position aware flight
    - the flow deck cannot be used to capture eagle eye camera shots
- research drone 2 drone communication
    - wifi and bluetooth are not viable
    - p2p communication is viable via radio
- research drone rotation to scan the environment
    - this is a viable option using gaussian splatting algorithms to convert 2D images to a 3D environment

### Todo

(deliverables)

- single non flying drone image stream to central server, existing object detection on central server
- single flying drone image stream to central server, map making on central server
- single non flying drone image stream to central server, position inference on central server
- make telemetry report of flying drone from origin, moves, to origin (check kalman)
- further experiments on P2P with 2 or 2+ drones
- interface that shows the drone path

### client feedback

- use a wooden pole, black and white pattern as a navigational element
- research what are hardware options for inference on the drone itself

## week 02

### Progress

- there is an api where you can upload images captured from the drone, these images get stitched together to form a pointcloud
- we have a router to create a network wifi network to stream these images, every drone gets a static ip address
- there is progress on drone collision avoidance using a P2P radio connection using a hardcoded predetermined position for each drone
- the flow deck is viable even on a sloped surface, we discovered this after testing in the old swimming pool
- the flow deck is can get confused by floor texture, we discovered this by testing in our working environment
- small deviations from course can result in a large accumulative error
- The gap between drones for collision avoidance should be bigger, 0.5m is not enough
- right now using drone net for detecting obstacles, it is an optimized model but for older hardware
- drone to environment collision avoidance appears difficult because the repositories are outdated
- yolo26n retrained with labeled images taken with the ai deck camera
- lighting can mess with the camera


### Todo

- create a pointcloud map from the images of multiple drones
- we need to think on how to integrate all the different sensors and systems (ai deck camera, flow deck, positioning, mapping)
- collision avoidance using a P2P radio connection needs to be improved and extended to more then 2 drones
- combining ai deck map navigation position inference and flow deck position estimate
- look into whole application architecture also including pre run necessities and communicating systems, delays during runs
- explore multiranger deck possibilities
- for drone to environment collision avoidance we need an environment that has more obstacles
- look into compression techniques for models for drone to environment collision avoidance and mapping
- try using segmentation or depth models for drone to environment collision avoidance
- need a way to remember previously detected items

### Client feedback

- track progress with incremental steps, focus on documentation
- detect drones and know which drone is being detected, know your nearest neighbour
- use homeing beacons (an object to recognize for the ai deck), this is redundant because of the mapping system
- speed is not important for this project, drones can go slowly
- document what would be needed for on board processing to detect obstacles (try raspberry pi)

### Extra

- we now have the ranger deck available
