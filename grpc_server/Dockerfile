# Use a smaller base image, such as Alpine Linux
FROM python:3.10-slim

# Set the working directory
WORKDIR /service

# Copy only the necessary files
COPY protobufs/ /service/protobufs/
COPY inference/ /service/inference/

# Install dependencies and remove temporary files in a single layer
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y build-essential && \
    python -m pip install --upgrade pip && \
    python -m pip install -r /service/inference/requirements.txt && \
    python -m grpc_tools.protoc -I /service/protobufs --python_out=. --grpc_python_out=. /service/protobufs/inference.proto && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Expose the required port
EXPOSE 50051

# Define the entry point
ENTRYPOINT [ "python", "/service/inference/inference.py" ]