
# %env OPENAI_API_KEY=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

"""##### import the relevant libraries"""

import streamlit as st
import openai
import os
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from PIL import Image

openai.api_key=st.secret["api_key"]


openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_image(input_string): 
  response = openai.Image.create(
    prompt=input_string,
    n=1,
    size="512x512"
  )
  image_url = response['data'][0]['url']
  return image_url

input_string = "a girl walking her dog in a field full of flowers"
output = generate_image(input_string)
print(f"Input: {input_string}\nOutput: {output}")

urllib.request.urlretrieve(output, 'output.png')
img = Image.open('output.png')
img_array = np.array(img)

plt.figure(figsize=(9,9))
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(img_array)

"""#### Generate Image Variation #####"""

openai.api_key = os.getenv("OPENAI_API_KEY")
def image_variation(image_file): 
  response = openai.Image.create_variation(
  image=open(image_file, "rb"),
  n=1,
  size="512x512",
  )
  image_url = response['data'][0]['url']
  return image_url

#load the image
image_file = '/content/drive/MyDrive/Colab Notebooks/CCS229/colab data/data image.png'
img = Image.open(image_file)

img_array = np.array(img)
plt.figure(figsize=(9,9))
ax = plt.axes(xticks=[], yticks=[])
ax.set_title("Original Image")
ax.imshow(img_array)

from google.colab import drive
drive.mount('/content/drive')

#generate the image variation

output = image_variation(image_file)
print(f"Output file URL: {output}")

urllib.request.urlretrieve(output, 'output.png')
plt.figure(figsize=(9,9))
img = Image.open('output.png')
img_array = np.array(img)

ax.set_title("Variation Image")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(img_array)
