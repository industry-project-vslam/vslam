# AI Deck Central

### Clone the repository

```shell
git clone https://github.com/bitcraze/aideck-gap8-examples.git

cd aideck-gap8-examples
```

### Copy files to the repository

|file|location in repository|
|---|---|
| [requirements.txt](./requirements.txt) | ./ |
| [yolo26n.pt](./yolo26n.pt) | ./ |
| [test.py](./test.py) | ./examples/other/wifi-img-streamer |

### Create a virtual environment

```shell
python3.12 -m venv .venv
```

### Activate the virtual environment Linux / MacOS
```shell
source .venv/bin/activate
```

### Activate the virtual environment Windows
```shell
.venv\Scripts\activate
```

### Install dependencies
```shell
pip install -r requirements.txt
```

### Run the object detection
```shell
python examples/other/wifi-img-streamer/test.py
```