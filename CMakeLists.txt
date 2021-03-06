cmake_minimum_required(VERSION 3.20)

set(CMAKE_TOOLCHAIN_FILE "vcpkg/scripts/buildsystems/vcpkg.cmake")
set(VCPKG_TARGET_TRIPLET "x64-windows-static")

cmake_policy(SET CMP0091 NEW)

project(Projet_OpenCV_Poker)

add_executable(Projet_OpenCV_Poker src/Main.cpp include/Image.h src/Image.cpp src/PokerAnalyzer.cpp include/PokerAnalyzer.h src/PokerCard.cpp include/PokerCard.h src/PokerTable.cpp include/PokerTable.h)

set_property(TARGET Projet_OpenCV_Poker PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

target_compile_options(Projet_OpenCV_Poker PRIVATE /Zc:__cplusplus)
target_compile_features(Projet_OpenCV_Poker PRIVATE cxx_std_20)

find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(Projet_OpenCV_Poker PRIVATE ${OpenCV_LIBS})

find_package(magic_enum CONFIG REQUIRED)
target_link_libraries(Projet_OpenCV_Poker PRIVATE magic_enum::magic_enum)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(Projet_OpenCV_Poker PRIVATE fmt::fmt)

add_custom_command(TARGET Projet_OpenCV_Poker PRE_BUILD
		COMMAND robocopy ${PROJECT_SOURCE_DIR}/resources/ ${PROJECT_BINARY_DIR}/resources /E /NS /NC /NFL /NDL /NP /NJH /NJS  || cmd /c "exit /b 0")
