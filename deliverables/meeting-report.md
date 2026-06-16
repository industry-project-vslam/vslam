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

## Week 02

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

## Week 03

### Progress

- mapping works perfectly for a single drone with an api image upload
- image processing for map making takes 30 seconds per image
- each drone is granted a static ip for a router network
- using set positions drones can communicate P2P for collision avoidance
- the drones can handle a sloped surface, the ranger deck does not get confused, the texture of the floor is the most important factor 
- testing of the error accumulation of the ranger deck resulted in an average of 10cm after a 1m² square flight sequence, but the takeoff and landing introduce even more error
- drones flying within 0.5m of eachother is dangerous for the drones due to error accumulation in the IMU
- using drone net to detect obstacles, this is not viable
- using the drone image stream we can use yolo26n for object detection
- discovered lighting has a big influence on the camera's performance

### Todo

- figuring out a way to stitch together maps to create a common map (multiple drones produce multiple maps that need to be merged)
- speed up image processing for map making
- infering position from drone image maps and a drone image
- getting started with the multi ranger deck for room mapping
- P2P collision avoidance with 3 drones
- creating a test environment for collision avoidance
- use depth models on the drone itself, try pulp tiny v3

### Client Feedback

- focus on documentation, track progress with incremental steps
- try to detect drones and know which drone is being detected, know your nearest neighbour
- use homeing beacons, an object to recognizere (this is redundant because of mapping)
- determine what would be needed for onboard object detection models
- look into the whole application architecture: pre run necessities, communicating systems, delays during runs
- remember previously detected items

## Week 04

### Progress

- state claasification implemented but performs badly due to drone image quality (it is a retrained mobilenet model)
- mapping different frames from different cameras is still difficult but works, in documentation all solutions have a camera that can detect depth, problem with combining drone relative position is depth estimation is not alligned
- ranger deck first, ai decks second approach can perform wall following with ranger decks and mapping safe zones, objects of interest and ai decks fly afterward
- ranger deck, ai decks grouped approach is difficult because of undetected objects that the following ai decks can crash into and needs more testing and a central compute system because of the limitted compute on the drone (192 kilo bytes)
- all drone research is stumped because of position estimation problems, limitted on board computation and the inexperience of firmware programming and lack of knowledge in navigational algorithms, a simulated testing environment could remove a lot of these drawbacks

### Todo

- we will try models where the drone position is part of the input parameters
- improving ranger deck first, ai decks second approach
- improving ranger deck, ai decks grouped approach
- finalise deliverables
    - visual demonstration of the ranger deck, ai decks grouped approach
    - integrate ranger deck first, ai decks second approach with image streaming and processing (object detection and pointcloud)

### Client feedback

- look into a human operated swarm, setting waypoints for drones to go to
- try using two radio's (this is mathematically impossible you need three)

## Week 05

### Todo

- demo filming
- state estimation model improvement
- documentation of successfull and failed approaches
- finalising all project management deliverables

