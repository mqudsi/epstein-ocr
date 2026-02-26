#!/bin/sh

if uname -r | grep -qi "microsoft"; then
    UV="uv.exe"
else
    UV="uv"
fi

$UV run monospace.py ../EFTA00400459-001.png ./train_top.txt ./train_bot.txt -d -q
