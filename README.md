# Remote portrait

Application for creating and manipulating of virtual remote portraits

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
python remote_portrait.py -i test_data/test.png -m_en models_ir/encoder_flame/E_flame.xml -m_flame models_ir/flame/flame.xml --template resources/head_template.obj
```

# Output
For now application output an .obj file with reconstructed 3d model of a head from single input image.