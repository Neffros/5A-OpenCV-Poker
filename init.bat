@echo off
set VCPKG_DEFAULT_TRIPLET=x64-windows-static
git clone https://github.com/Microsoft/vcpkg
cd vcpkg
git checkout 5ef52b5b75887fb150711f5effb221dd98b99e6f
call bootstrap-vcpkg.bat
vcpkg install opencv
rmdir /s /q .git
cd ..
pause