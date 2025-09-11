@echo off
cd /d S:\test\face
call .\.venv\Scripts\activate.bat
python watcher_service.py
