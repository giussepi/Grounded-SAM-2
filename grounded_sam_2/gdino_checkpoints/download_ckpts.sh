#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Define the URLs for the checkpoints
BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
ogc_file="groundingdino_swint_ogc.pth"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/${ogc_file}"
cogcoor_file="groundingdino_swinb_cogcoor.pth"
swinb_cogcoor_url="${BASE_URL}v0.1.0-alpha2/${cogcoor_file}"



# Download checkpoints using wget
if ! [ -f ./$ogc_file ]; then
    echo "Downloading ${ogc_file} checkpoint..."
    wget $swint_ogc_url || { echo "Failed to download checkpoint from $swint_ogc_url"; exit 1; }
else
    echo "${ogc_file} already exists; not retrieving"
fi

if ! [ -f ./$cogcoor_file ]; then
    echo "Downloading ${cogcoor_file} checkpoint..."
    wget $swinb_cogcoor_url || { echo "Failed to download checkpoint from $swinb_cogcoor_url"; exit 1; }
else
    echo "${cogcoor_file} already exists; not retrieving"
fi

echo "All checkpoints are downloaded successfully."
