# Control-By-Face
Control a robot by the shape of a face.

```
git clone https://github.com/sasekazu/control-by-face.git --recursive
```
or
```
git clone https://github.com/sasekazu/control-by-face.git --recursive
git submodule update --init --recursive
```

# patch for matplotlib-cpp
After clone, apply the patch to build matplotlib-cpp on windows:
```
cd (proj_dir)/matplotlib-cpp
patch -p0 < ../misc/fixforwin.patch
```