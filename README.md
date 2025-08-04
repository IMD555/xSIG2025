
## Devices

| Device | Model number |
|-------------|---------|
| PC | THIRDWAVE F-14LN5LA |
| Power meter | T3T-R4  |

## Environment

| Sofware | Version |
|-------------|---------|
| Ubuntu | 24.04.2 LTS |
| OpenVINO | 2025.0.0.17942 |

## Setup

https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/get-started.html

## Compile

```
cmake . && make
```

##  Run all test

```
bash test_matmul.sh
bash test_matmul_by_mul-RS.sh
bash test_bandwidth.sh
bash test_power.sh
```
