#!/bin/bash
docker run -it --rm \
          --name=tf_other \
          --network=host \
          -v ${PWD}:/tensorflow \
          lc6c/tensorflow:1.13.1 /bin/bash
          