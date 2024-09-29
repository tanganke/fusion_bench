# Docs: docs/guides/docker.md

# Use the official PyTorch image as the base image
ARG MIRROR=registry.dockermirror.com
FROM ${MIRROR}/pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel

# Set the working directory
WORKDIR /workspace

# Copy current directory contents into the container at /workspace/fusion_bench
COPY . /workspace/fusion_bench

# Install the required packages
RUN cd fusion_bench && pip install -e .

# Set the entrypoint to a bash shell
ENTRYPOINT ["/bin/bash"]
