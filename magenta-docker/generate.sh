#!/bin/bash

docker run --rm -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/models:/app/models" \
  magenta-minimal \
  python melody_rnn_generate \
    --config=basic_rnn \
    --bundle_file=/app/models/basic_rnn.mag \
    --output_dir=/app/output \
    --num_outputs=1 \
    --num_steps=128 \
    --primer_melody="[69, 71, 72, 74]" \
    --condition_on_primer=true \
    --inject_primer_during_generation=false \
    --temperature=1.0
