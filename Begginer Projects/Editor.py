from PIL import Image, ImageEnhance, ImageFilter
import os

path = './imgs'
pathout = '/editedImgs'

for filename in os.listdir(path):
    img = Image.open(f"{path}/{filename}")
    edit = img.filter(ImageFilter.SHARPEN).convert("L").rotate(-90)
    
    factor = 1.5
    enhancer = ImageEnhance.Contrast(edit)
    edit = enhancer.enhance(factor)
    
    clean_name = os.path.splitext(filename)
    
    edith.save(f.'{pathout}/{clean_name}_edit')
    