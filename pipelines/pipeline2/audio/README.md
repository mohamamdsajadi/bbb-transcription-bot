# How to get a valid ogg opus file

## Install ffmpeg
```bash
sudo apt update && sudo apt install ffmpeg -y
```

## Read the header of the file
### Make shure that the file is a valid ogg file
```bash
$ ffmpeg -i audio.ogg -f ffmetadata - 2>&1 | grep -A 1 "Input #0"
Input #0, ogg, from 'audio.ogg':
  Duration: 00:11:54.16, start: 0.000000, bitrate: 133 kb/s
```
If it does not show `Input #0, ogg` then the file is not a valid ogg file

### Make shure inside the ogg data there is a opus stream
```bash
$ ffmpeg -i audio.ogg -f ffmetadata - 2>&1 | grep -A 1 "Stream #0:0"
  Stream #0:0(eng): Audio: opus, 48000 Hz, stereo, fltp
    Metadata:
```
If it does not show `Audio: opus` then the file is not a valid ogg opus file