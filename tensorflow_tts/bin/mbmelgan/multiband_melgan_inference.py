#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import yaml
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow_tts.inference import AutoConfig
from tensorflow_tts.inference import TFAutoModel


# In[ ]:


mb_melgan = TFAutoModel.from_pretrained("tensorspeech/tts-mb_melgan-ljspeech-en")


# # Save to Pb

# In[ ]:


tf.saved_model.save(mb_melgan, "./mb_melgan", signatures=mb_melgan.inference)


# # Load and Inference

# In[ ]:


mb_melgan = tf.saved_model.load("./mb_melgan")


# In[ ]:


mels = np.load("../dump/valid/norm-feats/LJ001-0009-norm-feats.npy")


# In[ ]:


audios = mb_melgan.inference(mels[None, ...])


# In[ ]:


plt.plot(audios[0, :, 0])


# In[ ]:




