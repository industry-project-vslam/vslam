# AI Deck Central

```shell
git clone https://github.com/bitcraze/aideck-gap8-examples.git
cd aideck-gap8-examples
```

in the aideck-gap8-examples repository add the other files from this folder

|file|location in repository|
|---|---|
|requirements.txt|./|
|yolo26n.pt|./|
|test.py|./examples/other/wifi-img-streamer|

```shell
python3.12 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

```shell
python examples/other/wifi-img-streamer/test.py
```