# autodocking-ros2

To use this repository code you first need to download the `sensing-rigs-ros2` [repo](https://github.com/nautilus-unipd/sensing-rigs-ros2):
```bash
git clone https://github.com/nautilus-unipd/sensing-rigs-ros2 && \
cd sensing-rigs-ros2
```
---
TEMPORARY FIX

Switch to the `docker-fix` branch since the `main` branch downloads the latest Docker image (that has some errors):
```bash
git checkout docker-fix
```
---
Download this repository:
```bash
cd .. && \
git clone https://github.com/nautilus-unipd/autodocking-ros2
```

Download the model weights:

```bash
cd autodocking-ros2 && sudo apt install -y git-lfs && git lfs install && git lfs pull
```

Copy this repo to inside `sensing-rigs-ros2`:
```bash
cd .. && cp -r autodocking-ros2/autodocking_saver sensing-rigs-ros2/ros2_ws/src/sensing_nodes/
```
