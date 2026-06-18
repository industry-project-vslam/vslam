# Drone Square Flight Accuracy Test

## Objective
The goal of this test was to evaluate the positional accuracy of a drone while flying a predefined square flight path and returning to its starting position.

## Test Setup
To create an accurate reference for the experiment, tape and a ruler were used to measure a square path of 1 meter by 1 meter. The drone was programmed to:

1. Fly 1 meter forward  
2. Fly 1 meter to the left  
3. Fly 1 meter backward  
4. Return to the original starting position  

This setup allowed the final landing position of the drone to be compared with the actual starting point.

---

## First Test
In the first implementation, the drone completed the entire square path without landing at intermediate points. After finishing the route, it landed near the original starting position.

### Result
The flight was reasonably accurate, with the drone ending approximately **5 cm away** from the original starting point.

---

## Second Test
The code was then modified so that the drone would land at each corner of the square before taking off again and continuing to the next point.

### Result
This version produced significantly larger positioning errors. During landing and takeoff, the drone often slightly rotated or shifted because of contact with the ground. These small movements introduced additional inaccuracies into the flight path.

As a result, the drone sometimes ended up **10 cm or more away** from the expected position.

---

# Conclusion: Cumulative Error

The experiment demonstrates the effect of cumulative error in drone navigation. Small inaccuracies in movement, orientation, and positioning accumulate over time and become more noticeable after multiple actions.

In the first test, the drone only accumulated error while flying, which resulted in a relatively small deviation of around 5 cm. In the second test, however, every landing and takeoff introduced additional rotational and positional disturbances. These disturbances increased the overall error significantly.

This shows that repeated takeoff and landing sequences can negatively affect positional accuracy, especially when the drone relies mainly on onboard estimation and sensor data. Even small orientation changes at each corner can propagate throughout the remainder of the flight path, leading to larger final deviations from the intended position.