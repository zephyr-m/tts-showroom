#!/usr/bin/env bash
cd "$(dirname "$0")"
source "$(dirname "$0")/nix-libs.sh"
exec ./venv/bin/python "$@"
