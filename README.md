# VSLAM

## Kickoff notes

- Swarm sees and comunicates model of area to the operator
- (Procedural moving?)
- Swarm
    - distributed data processing
    - Human in the loop (directed autonomy)
    - Redundant, Adaptive
    - Battery (5-7min max)/range
    - Api data processing
    - Can drones estimate distance between each other?
- Architecture
    - Member
        - Preferably on board image processing
        - Swarm members have a line of sight
    - Human Swarm Interface (HMI)
        - Operator gets 3d model
        - Commands for controls
- Be able to estimate swarm size for a task
- Swarm layers
    - Sensors
    - Individual UAV
        - Reptilian brain for coordination
        - Desigion logic (e.g. if object is seen, do action) pattern → action
    - Subswarm
    - Swarm
    - Operator + swarm
- Applications
    - Infrastructure inspection
    - Rescue/search
    - Precision agriculture (each plant is treated individually)
- Routemap
    - FPV
    - FPV + object detection/annotation
    - FPV + auto actions
    - Fully auto actions
- Beacons for spacial orientation
- Create simulation
- Practical tests
- Questions
    - Pointcloud or gausian splatting? - Up to us
- Some drones have absolute position/some have visual
- Scope
    - 3d environment
    - Markers?
    - Detect objects/allow to move around them by commands
- Spaceial awareness

## Documentation

Check out our documentation [README.md](./docs/README.md)