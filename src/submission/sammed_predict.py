import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from segment_anything.build_sam3D import sam_model_registry3D

torch.set_grad_enabled(False)


def transform(image3D_np):
    image4D: torch.Tensor = torch.tensor(image3D_np).float().unsqueeze(0)
    transform = tio.Compose(
        [
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=lambda x: x > 0),
        ]
    )
    image5D = transform(image4D).unsqueeze(0)
    return image5D


def get_initial_corners(x):
    boxes = x["boxes"]
    corners: torch.Tensor = torch.zeros((len(boxes), 2, 3), dtype=torch.float32)

    for i, box in enumerate(boxes):
        corners[i, 0, :] = torch.tensor([box["z_min"], box["z_mid_y_min"], box["z_mid_x_min"]])
        corners[i, 1, :] = torch.tensor([box["z_max"], box["z_mid_y_max"], box["z_mid_x_max"]])

    return corners


def get_points(x, device):
    if "clicks" not in x.keys():
        return None

    num_clicks = len(x["clicks"][0]["fg"]) + len(x["clicks"][0]["bg"])
    points: torch.Tensor = torch.zeros((len(x["clicks"]), num_clicks, 3), dtype=torch.float32)
    click_types: torch.Tensor = torch.zeros((len(x["clicks"])), dtype=torch.float32)

    for i, click in enumerate(x["clicks"]):
        for j, point in enumerate(click["fg"] + click["bg"]):
            points[i, j, :] = torch.tensor(point)
            click_types[i] = 1 if j < len(click["fg"]) else 0

    return [points.to(device), click_types.to(device)]


def model_forward(model, image_embedding, low_res_masks, points, boxes, args):
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    prev_masks = F.interpolate(
        low_res_masks, size=(args.img_size, args.img_size, args.img_size), mode="trilinear", align_corners=False
    )
    return low_res_masks, prev_masks


def crop_image_on_boxes(image5D, corners, img_size):
    # if image5D is smaller than img size in any dimension, pad it
    paddings = {"x": 0, "y": 0, "z": 0}
    pad_mode = "constant"
    if image5D.shape[2] < img_size:
        pad = (img_size - image5D.shape[2]) // 2
        image5D = F.pad(image5D, (0, 0, 0, 0, pad, pad + 1), mode=pad_mode)
        paddings["z"] = pad
    if image5D.shape[3] < img_size:
        pad = (img_size - image5D.shape[3]) // 2
        image5D = F.pad(image5D, (0, 0, pad, pad + 1, 0, 0), mode=pad_mode)
        paddings["y"] = pad
    if image5D.shape[4] < img_size:
        pad = (img_size - image5D.shape[4]) // 2
        image5D = F.pad(image5D, (pad, pad + 1, 0, 0, 0, 0), mode=pad_mode)
        paddings["x"] = pad

    volume_crops = torch.zeros((len(corners), 1, img_size, img_size, img_size), dtype=torch.float32)
    crop_indices = torch.zeros((len(corners), 3), dtype=torch.int32)

    for i, box in enumerate(corners):
        z_min, y_min, x_min = box[0].int().tolist()
        z_max, y_max, x_max = box[1].int().tolist()

        z_middle = (z_min + z_max) // 2 if paddings["z"] == 0 else img_size // 2
        y_middle = (y_min + y_max) // 2 if paddings["y"] == 0 else img_size // 2
        x_middle = (x_min + x_max) // 2 if paddings["x"] == 0 else img_size // 2

        z_min = z_middle - img_size // 2
        y_min = y_middle - img_size // 2
        x_min = x_middle - img_size // 2

        z_max = z_middle + img_size // 2
        y_max = y_middle + img_size // 2
        x_max = x_middle + img_size // 2

        if z_min < 0:
            z_min = 0
            z_max = img_size
        if y_min < 0:
            y_min = 0
            y_max = img_size
        if x_min < 0:
            x_min = 0
            x_max = img_size

        if z_max > image5D.shape[2]:
            z_max = image5D.shape[2]
            z_min = image5D.shape[2] - img_size
        if y_max > image5D.shape[3]:
            y_max = image5D.shape[3]
            y_min = image5D.shape[3] - img_size
        if x_max > image5D.shape[4]:
            x_max = image5D.shape[4]
            x_min = image5D.shape[4] - img_size

        volume_crops[i, 0] = image5D[0, 0, z_min:z_max, y_min:y_max, x_min:x_max]
        crop_indices[i] = torch.tensor([z_min, y_min, x_min])

    return volume_crops, crop_indices


def place_pred_crop_in_volume(pred, pred_crop, crop_indices, i):
    z_min, y_min, x_min = crop_indices
    z_max = z_min + pred_crop.shape[2]
    y_max = y_min + pred_crop.shape[3]
    x_max = x_min + pred_crop.shape[4]

    if pred_crop.shape[2] > pred.shape[2]:
        pad = (pred_crop.shape[2] - pred.shape[2]) // 2
        pred_crop = pred_crop[:, :, pad : pad + pred.shape[2], :, :]
    if pred_crop.shape[3] > pred.shape[3]:
        pad = (pred_crop.shape[3] - pred.shape[3]) // 2
        pred_crop = pred_crop[:, :, :, pad : pad + pred.shape[3], :]
    if pred_crop.shape[4] > pred.shape[4]:
        pad = (pred_crop.shape[4] - pred.shape[4]) // 2
        pred_crop = pred_crop[:, :, :, :, pad : pad + pred.shape[4]]

    pred[i, 0, z_min:z_max, y_min:y_max, x_min:x_max] = pred_crop[0, 0]


def binarize_output(pred, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = torch.cat((torch.ones_like(pred)[0:1] * threshold, pred), dim=0)  # add "background" class
    pred = pred.argmax(dim=0).squeeze(0)

    return pred.cpu().numpy()


def save_prev_logits(low_res_pred_crops, args):
    parts = args.load_path.split(os.sep)
    prev_pred_path = os.sep.join(parts[:-1]) + os.sep + "prev_logits_" + parts[-1]
    low_res_pred_crops = torch.cat(low_res_pred_crops, dim=0)
    np.savez(prev_pred_path, prev_logits=low_res_pred_crops.cpu().numpy())


def predict(x, model, args):
    device = args.device

    image5D = transform(x["imgs"]).to(device)

    corners: torch.Tensor = get_initial_corners(x).to(device)
    volume_crops, crop_indices = crop_image_on_boxes(image5D, corners, args.img_size)
    volume_crops = volume_crops.to(device)
    crop_indices = crop_indices.to(device)
    points: torch.Tensor = get_points(x, device)

    if "prev_logits" not in x:
        prev_masks = torch.zeros_like(image5D).repeat(len(corners), 1, 1, 1, 1)
    else:
        prev_masks = x["prev_logits"]
    low_res_masks: torch.Tensor = F.interpolate(
        prev_masks.float(),
        size=(args.img_size // 4, args.img_size // 4, args.img_size // 4),
    )

    pred = torch.zeros_like(prev_masks) - 5
    low_res_pred_crops = []

    for i in range(len(corners)):
        embedding: torch.Tensor = model.image_encoder(volume_crops[i : i + 1])

        if points is not None:
            p = [points[0][i : i + 1] - crop_indices[i], points[1][i : i + 1]]
        else:
            p = points

        corners[i] -= crop_indices[i]

        low_res_pred_crop, pred_crop = model_forward(
            model,
            embedding,
            low_res_masks[i : i + 1],
            points=p,
            boxes=corners[i : i + 1],
            args=args,
        )
        place_pred_crop_in_volume(pred, pred_crop, crop_indices[i], i)
        low_res_pred_crops.append(low_res_pred_crop)

    save_prev_logits(low_res_pred_crops, args)

    binarized_pred = binarize_output(pred)

    import nibabel as nib

    lab = nib.Nifti1Image(binarized_pred.astype(np.float32), np.eye(4))
    nib.save(lab, os.path.join("pred.nii.gz"))

    img = image5D[0, 0].cpu().numpy()
    img = nib.Nifti1Image(img.astype(np.float32), np.eye(4))
    nib.save(img, os.path.join("img.nii.gz"))

    return binarized_pred


def load_data(file):

    parts = file.split(os.sep)
    boxes_path = os.sep.join(parts[:-1]) + os.sep + "boxes_" + parts[-1]

    x = np.load(file, allow_pickle=True)
    if "boxes" in x:
        np.savez(boxes_path, boxes=x["boxes"])
    else:  # read the boxes from the file
        boxes = np.load(boxes_path, allow_pickle=True)["boxes"]
        x = {**x, "boxes": boxes}

    prev_pred_path = os.sep.join(parts[:-1]) + os.sep + "prev_logits_" + parts[-1]
    if "prev_logits" in x:
        previous_logits = torch.tensor(np.load(prev_pred_path, allow_pickle=True)["prev_logits"])
        x = {**x, "prev_logits": previous_logits}

    return x


def load_predict_and_save(args):
    model = load_sam_model(args)

    file = [f for f in os.listdir(args.load_path) if (f.endswith(".npz") and "boxes_" not in f)][0]
    full_file = os.path.join(args.load_path, file)

    data = load_data(full_file)
    prediction = predict(data, model, args)

    save_file_path = os.path.join(args.save_path, file)
    np.savez(save_file_path, segs=prediction)


def load_sam_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sam_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    sam_model.eval()
    return sam_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the segmentation of the input images.")
    parser.add_argument("--load_path", type=str, help="Path to the input images.")
    parser.add_argument("--save_path", type=str, help="Path to save the predictions.")
    parser.add_argument("--model_type", type=str, default="vit_b_ori_norm", help="Model type to use for prediction.")
    parser.add_argument("--checkpoint", type=str, help="Path to the model weights.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the inference on.")
    parser.add_argument("--img_size", type=int, default=128, help="Size of the input image.")

    args = parser.parse_args()
    load_predict_and_save(args)
