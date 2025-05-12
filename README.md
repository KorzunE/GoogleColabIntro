#@title Необходимые функции
!pip install jax jaxlib
!pip -q install diffusers
!pip -q install transformers scipy ftfy accelerate
!pip -q install "ipywidgets>=7,<8"
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from google.colab import output
output.enable_custom_widget_manager()

stableDiffusion = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
stableDiffusion = stableDiffusion.to("cuda")


def createImagesStableDiffusion(prompt='', rows=2, cols=2, iteration=20):
  # Запускаем генерацию
  images =  stableDiffusion([prompt] * (rows*cols), num_inference_steps=iteration).images
  w, h = images[0].size
  grid = Image.new('RGB', size=(cols*w, rows*h))
  grid_w, grid_h = grid.size

  for i, img in enumerate(images):
      grid.paste(img, box=(i%cols*w, i//cols*h))
  display(grid)

# Изменяя текст в кавычках получайте различные изображени. Вам необходимо получить 5 разных генераций, которыми вы захотите поделиться с Куратором :)
createImagesStableDiffusion('Salvador Dali walks down the street with a cockroach on a leash, city, surrealism, crowded, people turn around, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)

# Изменяя текст в кавычках получайте различные изображени. Вам необходимо получить 5 разных генераций, которыми вы захотите поделиться с Куратором :)
createImagesStableDiffusion('Baby rats playing poker, realism, crowded, drink juice, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)

# Изменяя текст в кавычках получайте различные изображени. Вам необходимо получить 5 разных генераций, которыми вы захотите поделиться с Куратором :)
createImagesStableDiffusion('Cats sunbathing on the beach in thailand, on the sun beds, realism, sunny, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)

createImagesStableDiffusion('subway with people on the moon, people look considerate, cartoon, night, 8k, highly detailed, –ar 16:9 ', 2, 2, 100)

createImagesStableDiffusion('A robot cooks food in the kitchen for ancient people, realism, at least 10 people, 8k, highly detailed, –ar 16:9 ', 1, 2, 100)

createImagesStableDiffusion('unicycle racing on the seabed, surrealism, dogs, green water, 8k, highly detailed, –ar 16:9 ', 1, 3, 100)
