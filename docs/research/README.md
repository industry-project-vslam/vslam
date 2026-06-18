# Reasearch

In this document you can see our research that looks at the possiblities and limitations of the crazyflie system

## mo 18/05

- Capturing images with the flow deck will not be possible
- Combining the Loco positioning deck and the AI deck is not possible https://github.com/orgs/bitcraze/discussions/2178

## tu 19/05

- Drone using the flow deck will likely not fly stable on sloped surfaces https://store.bitcraze.io/collections/decks/products/flow-deck-v2
- Bluetooth cannot provide peer to peer drone communication https://www.bitcraze.io/documentation/repository/crazyflie2-nrf-firmware/master/protocols/ble/
- Peer to peer drone communication is possible using radio https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/p2p_api/
- There is a possibility of using image with AI to determine depth but we will need to retrain some of these models for grayscale images 