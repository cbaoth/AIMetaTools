@echo off

python %~dp0\main.py --mode EXTRACT --extract-field workflow %* | clip

:: pause
