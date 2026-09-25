## C++ Implementation – Build and Usage Instructions
 
The main program for IBF based alignment of logs to proxy logs is written in c++ so before the other scripts can be used, it needs to be compiled:


### Requirements

| Dependency | Version | Note |
|---|---|---|
| [GCC](https://gcc.gnu.org/) | ≥ 11 | other compilers (Clang/Apple Clang, MSVC) are not supported by SeqAn3 |
| [CMake](https://cmake.org/) | ≥ 3.16 | |
| [SeqAn3](https://github.com/seqan/seqan3) | 3.4.0 (commit `d4a7c88fd`) | **not** included in the repository, see [Installing SeqAn3](#installing-seqan3) |
| [cereal](https://github.com/USCiLab/cereal) | ≥ 1.3.1 | fetched automatically as a submodule of SeqAn3 |


The libraries [pugixml](https://pugixml.org/) (XML processing) and [MurmurHash3](https://github.com/aappleby/smhasher) (hashing) are included directly in the `source/` directory.

Tested on Linux (Ubuntu 24.04) with CMake 4.4.3 and GCC 13.3.0.


### Installing SeqAn3

This project uses **SeqAn3 3.4.0 (development snapshot)**, specifically commit `d4a7c88fd` on the `main` branch of the official repository.
To avoid compatibility issues, exactly this version should be used.

SeqAn3 is fetched into the existing (empty) `seqan3` directory next to `source/`, since `CMakeLists.txt` expects it there:

```bash
cd c++/index-basiertes-conformance-checking/seqan3
git init
git remote add origin https://github.com/seqan/seqan3.git
git fetch origin
git checkout d4a7c88fd
git submodule update --init --recursive
cd ..
```

Important: Check out the commit first, then initialize the submodules so that they match the correct version.
Because the directory contains a placeholder file (`.gitkeep`), a direct `git clone` into it does not work; hence the approach using `git init` and `git fetch`.

### Building

```bash
cd c++/index-basiertes-conformance-checking/build
cmake -DCMAKE_BUILD_TYPE=Release ../source
cmake --build .
```

The executables are then located in the `build/` directory.

#### Moving the Executables to `../scripts`

The Python scripts in `../scripts` expect the executables `alignment_txt` and `xes_to_txt` to be located in the `scripts/` directory.
After building, move them there from the repository root:

```bash
mv c++/index-basiertes-conformance-checking/build/alignment_txt scripts/
mv c++/index-basiertes-conformance-checking/build/xes_to_txt scripts/
```



### Programs

| Program | Description |
|---|---|
| `xes_to_txt` | Converts an event log in XES format to the text format. |
| `txt_to_xes` | Converts an event log in text format back to XES format. |
| `remove_double_traces` | Removes duplicate traces from an event log. |
| `alignment_txt` | Performs index-based conformance checking (alignment) on logs in text format. |


### Usage

The compiled programs (`xes_to_txt`, `alignment_txt`, ...) can be executed directly from the command line.

However, for better integration and automation, it is recommended to run them via Python using the `subprocess` module. 
This approach allows easier handling of inputs, outputs, and error management within a [standard workflow](../README.md#typical-workflow).

See in particular the [helper modules](../ARCHITECTURE.md#helper-modules) , e.g.: `../scripts/subprocess_wrapper.py`



##
#### Third-party libraries used:
- SeqAn3 – BSD-3-Clause
- pugixml – MIT
- MurmurHash3 – Public Domain

