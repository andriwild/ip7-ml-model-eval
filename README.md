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

### Compile to Hailo
```bash
hailomz compile \
    --hw-arch hailo8l \
    --yaml ./hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov5n_flower.yaml \
    --ckpt /local/shared_with_docker/flower_n.onnx \
    --classes 3 \
    --end-node-names /model.24/m.0/Conv /model.24/m.1/Conv /model.24/m.2/Conv \
    --calib-path /local/shared_with_docker/flowers/

hailomz compile \
    --hw-arch hailo8l \
    --yaml ./hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml \
    --ckpt /local/shared_with_docker/yolov8n_flower_v1.onnx \
    --classes 3 \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv \
    --calib-path /local/shared_with_docker/flowers/


hailomz compile \
    --hw-arch hailo8l \
    --yaml ./hailo_model_zoo/hailo_model_zoo/cfg/networks/yolov8n.yaml \
    --ckpt /local/shared_with_docker/yolov8n_pollinator_ep50_v1.onnx \
    --classes 5 \
    --end-node-names /model.22/cv2.0/cv2.0.2/Conv /model.22/cv3.0/cv3.0.2/Conv /model.22/cv2.1/cv2.1.2/Conv /model.22/cv3.1/cv3.1.2/Conv /model.22/cv2.2/cv2.2.2/Conv /model.22/cv3.2/cv3.2.2/Conv \
    --calib-path /local/shared_with_docker/pollinators/

```
Output Layer 8n:
Conv 39
Conv 42
Conv 50
Conv 53
Conv 60
Conv 63

## Coral USB

Ensure that the connected hohttps://github.com/pytorch/pytorch/issues/139052st device, such as a Raspberry Pi, is equipped with a reliable and high-capacity power supply.

## Coral Edge TPU

https://gist.github.com/dataslayermedia/714ec5a9601249d9ee754919dea49c7e

Example:
https://github.com/google-coral/tflite/blob/master/python/examples/detection/detect.py


## TFLite

Error in cpuinfo: prctl(PR_SVE_GET_VL) 
https://github.com/pytorch/pytorch/issues/139052


# Google Colab

Prevent from disconnecting:
```js
function ClickConnect() {
  console.log('Working')
  document
    .querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect')
    .click()
}
intervalTiming = setInterval(ClickConnect, 60000)
```
