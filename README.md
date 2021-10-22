# ordered-dithering-cpp
A tool for performing black and white Bayer dithering, written in C++17.

## Usage
```
# Use the 8x8 Bayer kernel (currently available sizes are 2x2, 4x4, and 8x8).
# Increase pre-dithering brightness by 30 (pixel intensity values range from 0-255).
# Increase pre-dithering contrast by 15%.
# For the PNG output format, a bit depth of 1 is automatically selected.
ordered_dithering_main /path/to/some_image.jpg /path/to/output.png 8 30 1.15
```

## Dependencies
There are a few dependencies required to build this tool:
#### Bundled/Fetched by CMake
* fmt (https://github.com/fmtlib/fmt)
#### Not included
* libvips (https://github.com/libvips/libvips)

## Building
The recommended way of building this project is as follows:
1. Grab the source code
```
git clone https://github.com/nilssonk/ordered-dithering-cpp
cd ordered-dithering-cpp
```
2. Create and enter the build directory
```
mkdir -p build
cd build
```
3. Run CMake to configure the project
```
# Debug build with clang-tidy static analysis enabled
cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug -DORDERED_DITHERING_CLANG_TIDY=ON ../
# Release build with LTO enabled
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DORDERED_DITHERING_CLANG_TIDY=OFF -DORDERED_DITHERING_LTO=ON ../
```
4. Build using your configured build system
```
ninja
```

If all goes well, the executable can then be found at *main/ordered_dithering_main* inside the build directory.
