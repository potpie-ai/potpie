#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pem_file>"
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Error: File $1 does not exist"
    exit 1
fi

awk '!/BEGIN|END/ && NF' "$1" | tr -d '\n'
echo
