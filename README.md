# dyna_remov  

Masking the dynamic object

-run the c++ file
```
g++ dyna_cpp.cc -o dyna_cpp $(pkg-config --cflags --libs opencv4) 
```
```
./dyna_cpp
```

run the python file
```
python3 dyna_python.py
```

## create and draw orb points

create and draw orb points on whole frame

```
g++ orb_1.cc -o orb_1 $(pkg-config --cflags --libs opencv4)

```
```
./orb_1
```

create adn draw orb points excluding the points in dynamic region (masked one)

```
g++ orb_2.cc -o orb_2 $(pkg-config --cflags --libs opencv4)
```
```
./orb_2
```
