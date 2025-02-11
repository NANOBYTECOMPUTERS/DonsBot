@echo off
cd models
yolo export model=1.pt format=engine half=False
