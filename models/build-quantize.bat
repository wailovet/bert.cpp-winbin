set INCLUDE=%~dp0..\ggml\include\ggml;%~dp0..\ggml\include;%~dp0..\;%~dp0 
@REM call "F:\VisualStudio\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
 
cl.exe /EHsc /arch:AVX2 /Ot /Ox /Gs /std:c++latest   %~dp0..\ggml\src\ggml.c %~dp0..\bert.cpp quantize.cpp
   
echo "ok" 
pause