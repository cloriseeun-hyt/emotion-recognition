@echo off
setlocal

set "PROJECT_DIR=e:\emotion-recognition"
set "PYTHON_EXE=%PROJECT_DIR%\venv\Scripts\python.exe"
set "APP_FILE=%PROJECT_DIR%\frontend\app.py"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] 未找到虚拟环境 Python：%PYTHON_EXE%
    echo 请先创建虚拟环境并安装依赖。
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo [ERROR] 未找到前端入口文件：%APP_FILE%
    pause
    exit /b 1
)

echo 正在启动 Emotion Recognition System...
echo.
echo 项目目录: %PROJECT_DIR%
echo Python: %PYTHON_EXE%
echo 应用入口: %APP_FILE%
echo.
echo 启动后浏览器若未自动打开，请手动访问：
echo http://localhost:8501
echo.

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" -m streamlit run "%APP_FILE%"

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Streamlit 启动失败，退出码：%EXIT_CODE%
    echo 你也可以手动执行下面这条命令排查：
    echo "%PYTHON_EXE%" -m streamlit run "%APP_FILE%"
    echo.
    echo 如果仍失败，请把完整报错发给我。
    pause
)

endlocal
