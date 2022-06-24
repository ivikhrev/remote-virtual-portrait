# Remote portrait
![](extended_example.gif)
Application for creating and manipulating of virtual remote portraits using DECA model. Application translate 3D scene with avatar to the virtual web-camera if it is enabled.

# Preparing to Run
The demo dependencies should be installed before run. That can be achieved with the following commands on Windows:

```
python -m venv remote-portrait
.\remote-portrait\Scripts\activate
pip install -r requirements.txt
```

# Running

Run the application with the next command line:

```
python remote_portrait.py
```

# Output
Application outputs an .obj file with reconstructed 3d model of a head from single input image and window with rendered 3d model