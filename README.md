# live-translation

## Pipeline
### Controller 1
### Phase: Preparing Data
- [ ] Create 10s audio clips
- [ ] Add data to a AudioData class
- [ ] Convert audio data to

### Controller 2
### Phase: Score Data
- [ ] Score the audio data

#### Phase: Detect speech
- [ ] Route data depending on the score
- [ ] Detect if someone is speaking VAD

#### Phase Process audio data
- [ ] Convert audio to Text

#### Phase: Clean data
- [ ] Remove text whis has a low confidence score
- [ ] Add next word prediction to the text

#### Phase: Translate
- [ ] Translate text to another language

## Prepare the Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Create a Ramdisk in Linux
To create a ramdisk in Linux with a size of 500MB and mount it at `/mnt/ramdisk/`, you can follow these steps:

1. **Create the mount point directory** (if it doesn't exist already):
   ```bash
   sudo mkdir -p /mnt/ramdisk
   ```

2. **Create and mount the ramdisk:**
   You can use the `tmpfs` filesystem to create a ramdisk. The following command will mount a 500MB ramdisk at `/mnt/ramdisk/`:
   ```bash
   sudo mount -t tmpfs -o size=500M tmpfs /mnt/ramdisk/
   ```

3. **Verify the mount:**
   You can check if the ramdisk is correctly mounted by using the `df -h` command:
   ```bash
   df -h | grep ramdisk
   ```

   This should show a line indicating that `/mnt/ramdisk` is mounted with a size of 500MB.

### Optional: Make the Ramdisk Persistent Across Reboots (Not Recommended)

Since ramdisks are typically volatile (they lose their contents after a reboot), if you want this to be mounted automatically after a reboot (although it's usually not recommended for a ramdisk due to its transient nature), you can add the following line to your `/etc/fstab`:

```bash
tmpfs   /mnt/ramdisk   tmpfs   defaults,size=500M   0   0
```

This way, the ramdisk will be created and mounted automatically during the boot process.

### To unmount the ramdisk:

If you want to unmount the ramdisk, you can use the following command:

```bash
sudo umount /mnt/ramdisk
```

That's it! You now have a 500MB ramdisk mounted at `/mnt/ramdisk/`.