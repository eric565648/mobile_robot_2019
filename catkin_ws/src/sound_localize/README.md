# Sound Localization

## On Robot
### Sound Localization
```
rosrun sound_localization server_main.py
```

## On Respeaker
### Sound Record
```
cd subt
sudo python3 client.py
```

## Dependency on Respeaker

- cmake
```
sudo apt install build-essential cmake
```

- pixel_ring: This library allow users to control the leds on respeakers.
```
git clone --depth 1 https://github.com/respeaker/pixel_ring.git
cd pixel_ring
sudo pip3 install -U -e . --user
```

- mraa
```
sudo apt install python3-mraa python3-upm libmraa1 libupm1 mraa-tools
```

- scipy
```
sudo apt-get install python3-scipy
```
