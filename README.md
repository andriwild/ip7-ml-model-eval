# Run the benchmark

Running the benchmark requires the following steps:
```bash
source ./benchmark_***.sh
```

Note: Do not run the script without sourcing it.
This is because the script sets environment variables that are used by the benchmarking tool.


# Known issues

## Hailo

Import error: `libGL.so.1` library:
```bash
apt-get install -y libgl1-mesa-dev
```

## Coral USB

Ensure that the connected hohttps://github.com/pytorch/pytorch/issues/139052st device, such as a Raspberry Pi, is equipped with a reliable and high-capacity power supply.

## Coral Edge TPU

https://gist.github.com/dataslayermedia/714ec5a9601249d9ee754919dea49c7e

Example:
https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect.py


## TFLite

Error in cpuinfo: prctl(PR_SVE_GET_VL) 
https://github.com/pytorch/pytorch/issues/139052

