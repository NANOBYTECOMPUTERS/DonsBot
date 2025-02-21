@echo off
cd models
yolo export model=1.pt imgsz=640 format=engine half=True
