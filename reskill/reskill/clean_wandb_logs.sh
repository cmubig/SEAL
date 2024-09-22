#!/bin/bash

for file in wandb/**/logs/debug-internal.log; do
    echo $file
done