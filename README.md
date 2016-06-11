# Dali examples

### Installing dependencies

##### Ubuntu

```bash
sudo apt-get install protobuf* libprotobuf* libsqlite*
```

##### Fedora

Tested on Fedora 23

```bash
sudo dnf install gflags gflags-devel protobuf protobuf-c protobuf-c-compiler protobuf-devel protobuf-c-devel sqlite sqlite-devel
```

### Compiling with local build of Dali

```bash
mkdir build
cd build
DALI_HOME=/path/to/Dali cmake ..
make -j9
```
