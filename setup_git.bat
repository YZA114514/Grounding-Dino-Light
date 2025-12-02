@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo 正在初始化 Git 仓库...
git init

echo 正在添加远程仓库...
git remote add origin https://github.com/YZA114514/Grounding-Dino-Light.git

echo 正在添加所有文件...
git add .

echo 正在提交文件...
git commit -m "Initial commit: Add GroundingDINO project files"

echo 正在推送到远程仓库...
git branch -M main
git push -u origin main

echo 完成！
pause

