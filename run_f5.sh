#!/bin/sh
cd /Users/queclay/Documents/MSDS/DS5001/encyclicals
python output/rebuild_f5.py > /tmp/f5_run.log 2>&1
echo "Exit code: $?"
