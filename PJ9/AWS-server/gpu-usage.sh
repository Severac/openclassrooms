#!/bin/bash
nvidia-smi -q |grep -B 4 -A 4 Utilization
