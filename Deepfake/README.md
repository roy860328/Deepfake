# Deepfake

## Requirements

```
Keras
face_recognition
h5py
```

## bug
AttributeError: module 'tensorflow.python.keras.backend' has no attribute 'get_graph'
```
pip install Keras==2.1.5
```

### load model
KeyError: "Can't open attribute (can't locate attribute: 'weight_names')"

h5py版本對不上keras問題
```
load_model(path + ".h5", custom_objects={'custom_loss_function': self.custom_loss_function})
```
to
```
load_weights(path + ".h5")
```
