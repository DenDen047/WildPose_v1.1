import cv2
import numpy as np
import os
import json
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything import SamPredictor

from config import COLORS, colors_indices



def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

data_dir = 'data/lion_sleep'
output_dir = os.path.join(data_dir, 'masks_lion')

IMG_WIDTH, IMG_HEIGHT = 1280, 720
OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720
FPS = 3
OUTPUT_VIDEO_PATH = os.path.join(data_dir, 'seg_lion_sleep.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, FPS,
                      (OUTPUT_WIDTH, OUTPUT_HEIGHT))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint='sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)
mask_predictor = SamPredictor(sam)

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(data_dir, 'train.json'), 'r') as f:
    gt_json = json.load(f)
img_dict = {}

annotations = gt_json['annotations']
images = gt_json['images']

# load annotation data
for idx in range(len(annotations)):
    anno = annotations[idx]
    bbox = anno['bbox']
    img_id = anno['image_id']
    obj_id = anno['category_id']

    if img_id not in img_dict:
        img_dict[img_id] = []
    img_dict[img_id].append([bbox, obj_id])

for idx in tqdm(range(len(images))):
    # if idx > 100:
    #    break
    # if idx < 7:
    #    continue
    img_fpath = os.path.join(data_dir, images[idx]['file_name'])
    img = cv2.imread(img_fpath)
    img_draw = np.copy(img)

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)
    bboxes = []
    objs = []
    masks_list = []
    if idx in img_dict:
        # cv2.putText(img, 'Frame Number: {0}'.format(idx), (20, 40), font, 1, (255,255,255), thickness, cv2.LINE_AA)
        for obj_idx in range(len(img_dict[idx])):
            bbox, obj_id = img_dict[idx][obj_idx]

            x1, y1, width, height = bbox
            x1, y1, width, height = int(x1), int(y1), int(width), int(height)
            x2, y2 = x1 + width, y1 + height
            bboxes.append([x1, y1, x2, y2])
            objs.append(obj_id)

            bbox_form = np.array([x1, y1, x2, y2])
            masks, scores, logits = mask_predictor.predict(
                box=bbox_form,
                multimask_output=True
            )
            mask_max = masks[np.argmax(scores)]
            color_RGB = COLORS[colors_indices[obj_id]]['color']
            color_BGR = [color_RGB[2], color_RGB[1], color_RGB[0]]
            masks_list.append(mask_max[None, :, :])
            img_draw = overlay(img_draw, mask_max, color_RGB, 0.5, resize=None)
            cv2.rectangle(img_draw, (x1, y1), (x1 + width, y1 + height),
                          (color_BGR[0], color_BGR[1], color_BGR[2]), 2)
            cv2.rectangle(img_draw, (x1, y1 - 30), (x1 + 50, y1),
                          (color_BGR[0], color_BGR[1], color_BGR[2]), 2)
            cv2.putText(
                img_draw,
                str(obj_id),
                (x1 + 5, y1 - 5),
                font,
                fontScale,
                tuple(color_BGR[0:2]),
                thickness,
                cv2.LINE_AA)
    np.save(
        os.path.join(output_dir, '{0}.npy'.format(idx)),
        masks_list)
    np.save(
        os.path.join(output_dir, '{0}_obj_ids.npy'.format(idx)),
        np.array(objs))

    out.write(img_draw)

out.release()
