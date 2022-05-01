![Cover](res/img/cover.png)

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=24&color=d11060&center=true&vCenter=true&width=640&height=24&lines=An+easy+to+use+library+for+neural+networks!;Provides+a+plethora+of+handy+functionality!">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=F0F0F0">

   <img src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=0A0A0A">

   <img src="https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0">
</p>

<p align="center">
   <a href="LICENSE.txt">
      <img alt="GitHub license" src=".github/BADGES/license.svg">
   </a>
</p>

## Overview

An open-source and cross-platform software library for working with neural networks that accentuates simplicity
and ease of use.

### Links

- [Repository](https://github.com/mansoormemon/cbrainx)

## Author

Mansoor Ahmed Memon

- [github.com/mansoormemon](https://github.com/mansoormemon)

## Getting Started

### Build Instructions

#### 0. Prerequisites

Before moving on to the build process, ensure the following prerequisites:

- [Git](https://git-scm.com/)
- [CMake](https://cmake.org/) (Required)

**Compilers:**

1. GCC (>= 10.0)
2. Clang (>= 10.0)
3. MSVC (>= 2019)

#### 1. Clone the repository

Clone the repository to your local machine using `git`.

```bash
git clone --recursive https://github.com/mansoormemon/cbrainx.git
```

Now, `cd` to the `cbrainx` project directory:

```bash
cd cbrainx
```

#### 2. Configure and Build

The following options are available for controlling the build process:

1. `CMAKE_INSTALL_PREFIX` - tells `CMake` to use this as the installation root directory.
    - **Default:**
        - `/usr/local/` - for Unix based platforms (including macOS).
        - `C:/Program Files (x86)/cbrainx` - for Windows platforms (both x86 and x64).

2. `CMAKE_BUILD_TYPE`
    - **Default:** `Release`.

3. `BUILD_SHARED_LIBS` - tells `CMake` to create a shared library.
    - **Default:** `OFF`.

4. `CBRAIN_BUILD_EXAMPLES` - tells `CMake` to build **examples**.
    - **Default:**
        - `ON` - when configured as a standalone project.
        - `OFF` - when configured as a dependency or a sub-project.

These options can be configured as desired.

Now, to build the project, run:

```bash
cmake -Bbuild
cmake --build build
```

#### 3. Installation

The installation process only copies the files to predefined locations as follows:

| Target         | Location                                  |
| -------------- | ----------------------------------------- |
| Public headers | `${CMAKE_INSTALL_PREFIX}/include/cbrainx` |
| Library        | `${CMAKE_INSTALL_PREFIX}/lib/cbrainx`     |

Finally, to install the project, run:

```bash
cmake --install build
```

## License

This product is distributed under the [Apache License, v2.0](https://www.apache.org/licenses/LICENSE-2.0).
