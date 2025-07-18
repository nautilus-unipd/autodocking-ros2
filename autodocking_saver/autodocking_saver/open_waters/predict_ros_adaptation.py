import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch


from autodocking_saver.open_waters.wasr import models as models

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)


# Get the absolute path of the current script file
script_path = os.path.abspath(__file__)

# Get the directory containing the current script
script_dir = os.path.dirname(script_path)


BATCH_SIZE = 4
MODEL_SIZE = "50"
#qui puoi cambiare l'architettura [cambia solo il numero (50 o 101)]
ARCHITECTURE = f'wasr_resnet{MODEL_SIZE}'
#se cambi l'architettura devi cambiare anche il path dei pesi di conseguenza
WEIGHTS_PATH = f"{script_dir}/weights/wasr_rn{MODEL_SIZE}.pth"
#questo e' il path del file in output
#OUT_PATH = "output/predictions/out.jpg"

def predict_image(model, image, imu_mask=None):

    feat = {'image': image.cuda() if torch.cuda.is_available() else image.to("cpu")}

    res = model(feat)
    prediction = res['out'].detach().softmax(1).cpu()
    return prediction


def predict(image : Image, save_image, image_path):
    #questo e' il path del file in output
    OUT_PATH = image_path
    #OUT_PATH = os.path.join(OUT_PATH, image_name)
    
    # Load and prepare model
    model = models.get_model(ARCHITECTURE, pretrained=True)

    state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')

    if 'model' in state_dict:
        # Loading weights from checkpoint
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)

    # Enable eval mode and move to CUDA if gpu is enabled
    model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

    # Load and normalize image
    img = np.array(image.resize((512, 288)).convert('RGB'))

    H,W,_ = img.shape
    img = torch.from_numpy(img) / 255.0
    img = (img - NORM_MEAN) / NORM_STD
    img = img.permute(2,0,1).unsqueeze(0) # [1xCxHxW]
    img = img.float()

    # Load IMU mask if provided
    imu_mask = None

    # Run inference
    probs = predict_image(model, img, imu_mask)
    probs = torch.nn.functional.interpolate(probs, (H,W), mode='bilinear')
    preds = probs.argmax(1)[0]

    # Convert predictions to RGB class colors
    preds_rgb = SEGMENTATION_COLORS[preds]
    preds_img = Image.fromarray(preds_rgb)

    if not save_image:
        return

    output_dir = Path(OUT_PATH).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    preds_img.save(OUT_PATH)


def main():
    # questo e' un esempio , Image.open(PATH) e' l'immagine passata con il path.
    predict(Image.open("examples/resized/res.png"))

if __name__ == '__main__':
    main()
