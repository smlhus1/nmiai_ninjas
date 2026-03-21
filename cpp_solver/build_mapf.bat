@echo off
REM Build C++ MAPF planner with MSVC 2019 BuildTools
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
cl /EHsc /O2 /std:c++17 /MT mapf.cpp /Fe:mapf.exe
