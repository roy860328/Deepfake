# Deepfake

## Requirements

```
Keras==2.1.5
tensorflow-gpu==1.8.0
mtcnn==0.1.0
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

### loss nan
取log的數值過小或過大，造成bug，透過clip，確保不會overflow
```
K.clip(y_predict, 1e-3, 1 - 1e-3)
```
