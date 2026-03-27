#!/bin/bash
# Git credential helper that reads token from GH_TOKEN environment variable
case "$1" in
get)
  echo "protocol=https"
  echo "host=github.com"
  echo "username=oauth2"
  echo "password=$GH_TOKEN"
  ;;
esac
