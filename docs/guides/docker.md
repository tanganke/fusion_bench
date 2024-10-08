# Build and Run Fusion Benchmarks in a Docker Container

## Build Image

Build the image using the following command:

```bash
# Using the mirror
docker build -t fusion_bench .
```

If you want to use the default Docker Hub, you can omit the MIRROR argument or set it to an empty value:

```bash
# Using the default Docker Hub
docker build --build-arg MIRROR=docker.io -t fusion_bench .
```

## Run Container

Test the container using the following command:

```bash
docker run --gpus all -it --rm fusion_bench
```
