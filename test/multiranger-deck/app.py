from services.swarm import SwarmService
import time

def run():
    swarm = SwarmService()
    try:
        last = time.time()
        count = 0
        while True:
            swarm.take_step()
            count += 1
            now = time.time()
            if now - last >= 1.0:
                count = 0
                last = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        close(swarm)


def close(swarm: SwarmService):
    try:
        swarm.close_links()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    run()