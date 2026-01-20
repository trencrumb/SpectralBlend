# Spectral Blend

Uses AudioTransport from [Flucoma](https://github.com/flucoma/flucoma-core) 
which implements [optimal transport](https://towardsdatascience.com/optimal-transport-a-hidden-gem-that-empowers-todays-machine-learning-2609bbf67e59) on the spectrum of two sources.

## Usage

Route Sound A to channels 1/2 and Sound B to channels 3/4 on the plugin in the DAW.

## Installation

Installation using the pre-built binaries by using the installers place the plugins in your system-wide plugin directories.
*e.g.* `/Library/Audio/Plug-Ins/*pluginformat*` on macOS.

### macOS

Tested on macOS 15.6.1 (Apple Silicon).

1. Download the latest release from the [Releases](https://github.com/trencrumb/SpectralShift/releases) page.
2. Open the `.pkg` installer and follow the prompts.
3. The build is a universal binary and supports both Intel and Apple Silicon Macs.

### Linux

Tested on Ubuntu 24.04 (x86_64).

1. Download the latest release from the [Releases](https://github.com/trencrumb/SpectralShift/releases) page.
2. Extract the archive.
3. For a quick install, run:

   ```bash
   ./install.sh -y
   ```
4. To uninstall:

   ```bash
   ./uninstall.sh
   ```

### Windows

Tested on Windows 25H2 (x86_64).

1. Download the latest release from the [Releases](https://github.com/trencrumb/SpectralShift/releases) page.
2. Run the `.msi` installer and follow the instructions.
3. The installer is not code signed, so Windows may show a warning about an unknown publisher. This is expected.

## Building from Source

### Prerequisites

#### All platforms

* CMake 3.24 or newer
* Git
* C++20-compatible compiler

#### macOS

* Xcode (latest stable recommended)
* Ninja (optional but much faster build times):

  ```bash
  brew install ninja
  ```

#### Linux (Ubuntu / Debian)

* GCC or Clang with C++20 support
* Ninja (optional):

  ```bash
  sudo apt install ninja-build
  ```
* JUCE dependencies:

  ```bash
  sudo apt-get update && sudo apt install \
    libasound2-dev \
    libx11-dev \
    libxinerama-dev \
    libxext-dev \
    libfreetype6-dev \
    libwebkit2gtk-4.0-dev \
    libglu1-mesa-dev
  ```

#### Windows

* Visual Studio 2019 or newer (Desktop development with C++)
* CMake
* Ninja (optional, via Chocolatey):

  ```powershell
  choco install ninja
  ```

### Build Instructions

```bash
git clone https://github.com/trencrumb/SpectralShift.git
cd SpectralShift
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --config Release
```

If `COPY_PLUGIN_AFTER_BUILD` is enabled, plugins are copied to user plugins directory after build.
*e.g.* `~/Library/Audio/Plug-Ins/*pluginformat*/`

Built plugins are also found in:

```
build/SpectralShift_artefacts/Release/
```