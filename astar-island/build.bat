@echo off
REM Build Astar Island simulator with MSVC
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
cl /EHsc /O2 /std:c++17 /MT /openmp sim.cpp /Fe:sim.exe
