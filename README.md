# Control-By-Face
Control a robot by the shape of a face.

```bash
git clone https://github.com/sasekazu/control-by-face.git --recursive
```
or
```bash
git clone https://github.com/sasekazu/control-by-face.git --recursive
git submodule update --init --recursive
```

# Collector

# Trainer
You might have to call trainer after environmental variables declaration.
```bash
set PYTHONHOME=/path/to/python
set PYTHONPATH=/path/to/python/lib
./traner
```
`/path/to/python` could be found by calling `where python` in windows command prompt or `which python` in bash.

# patch for matplotlib-cpp
After clone, apply the patch to build matplotlib-cpp on windows:
```
cd (proj_dir)/matplotlib-cpp
patch -p0 < ../misc/fixforwin.patch
```