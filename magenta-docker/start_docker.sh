#!/bin/bash

docker run -dit \
  --name my-magenta-container \
  -v "$(pwd)/models":/app/models \
  -v "$(pwd)/output":/app/output \
  magenta-minimal

