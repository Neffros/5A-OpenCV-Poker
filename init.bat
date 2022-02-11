@echo off
set VCPKG_DEFAULT_TRIPLET=x64-windows-static
git clone https://github.com/Microsoft/vcpkg
cd vcpkg
git checkout 6ba505cf2c1752d8ea5abb21427e23ff89dc486f
call bootstrap-vcpkg.bat
vcpkg install opencv fmt magic-enum
rmdir /s /q .git
cd ..
pause