import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt


USE_KEY = 'left'
NOT_USE_KEY = 'right'


def classify_images_and_save_to_csv(image_directory, train_csv_file, output_csv_file):
  print(f'Classifying images in {image_directory} and saving to {output_csv_file}')
  
  df = pd.read_csv(train_csv_file)

  df['use'] = None

  def on_key(event):
    if event.key == USE_KEY:
      set_label(1)
    elif event.key == NOT_USE_KEY:
      set_label(0)
  
  def set_label(label):
    nonlocal index
    df.at[index, 'use'] = label
    save_results()
    index += 1
    if index >= len(df):
      plt.close()
    else:
      show_image(index)
  
  def show_image(index):
    image_name = df.at[index, 'image_name']
    label = df.at[index, 'label']
    image_path = os.path.join(image_directory, image_name)
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f'{image_name} (label: {label})')
    plt.axis('off')
    plt.draw()
  
  fig, ax = plt.subplots()
  index = 0
  show_image(index)
  fig.canvas.mpl_connect('key_press_event', on_key)

  def save_results():
    df_fixed = df[df['use'] == 1]
    df_fixed.to_csv(output_csv_file, index=False)
  
  plt.show()

  print(f'Classified {len(df)} images and saved to {output_csv_file}')


if __name__ == '__main__':
  image_directory = os.path.join(os.path.dirname(__file__), "./data/input/images/train")
  train_csv_file = os.path.join(os.path.dirname(__file__), "./data/input/train.csv")
  output_csv_file = os.path.join(os.path.dirname(__file__), "./data/input/train_fixed.csv")
  
  classify_images_and_save_to_csv(image_directory, train_csv_file, output_csv_file)