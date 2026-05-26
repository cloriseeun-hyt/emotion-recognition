@echo off
cd /d e:\emotion-recognition

mkdir data\train\angry 2>nul
mkdir data\train\happy 2>nul
mkdir data\train\sad 2>nul
mkdir data\train\surprise 2>nul
mkdir data\train\neutral 2>nul
mkdir data\train\fear 2>nul
mkdir data\train\disgust 2>nul

mkdir data\test\angry 2>nul
mkdir data\test\happy 2>nul
mkdir data\test\sad 2>nul
mkdir data\test\surprise 2>nul
mkdir data\test\neutral 2>nul
mkdir data\test\fear 2>nul
mkdir data\test\disgust 2>nul

echo Directory structure created successfully!
