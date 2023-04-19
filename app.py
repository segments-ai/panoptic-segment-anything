import subprocess, os, sys

result = subprocess.run(["pip", "install", "-e", "GroundingDINO"], check=True)
print(f"pip install GroundingDINO = {result}")

result = subprocess.run(["pip", "install", "gradio==3.27.0"], check=True)
print(f"pip install gradio==3.27.0 = {result}")

sys.path.insert(0, "./GroundingDINO")

if not os.path.exists("./sam_vit_h_4b8939.pth"):
    result = subprocess.run(
        [
            "wget",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        ],
        check=True,
    )
    print(f"wget sam_vit_h_4b8939.pth result = {result}")


import argparse
import random
import warnings
import json

import gradio as gr
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy import ndimage
from PIL import Image
from huggingface_hub import hf_hub_download
from segments.utils import bitmap2file

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict,
)
from GroundingDINO.groundingdino.util.inference import annotate, predict

# segment anything
from segment_anything import build_sam, SamPredictor

# CLIPSeg
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation


def load_model_hf(model_config_path, repo_id, filename, device):
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    model = model.to(device)
    return model


def load_image_for_dino(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dino_image, _ = transform(image, None)
    return dino_image


def dino_detection(
    model,
    image,
    image_array,
    category_names,
    category_name_to_id,
    box_threshold,
    text_threshold,
    device,
    visualize=False,
):
    detection_prompt = " . ".join(category_names)
    dino_image = load_image_for_dino(image)
    dino_image = dino_image.to(device)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=dino_image,
            caption=detection_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )
    category_ids = [category_name_to_id[phrase] for phrase in phrases]

    if visualize:
        annotated_frame = annotate(
            image_source=image_array, boxes=boxes, logits=logits, phrases=phrases
        )
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
        visualization = Image.fromarray(annotated_frame)
        return boxes, category_ids, visualization
    else:
        return boxes, category_ids, phrases


def sam_masks_from_dino_boxes(predictor, image_array, boxes, device):
    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_array.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes_xyxy, image_array.shape[:2]
    ).to(device)
    thing_masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return thing_masks


def preds_to_semantic_inds(preds, threshold):
    flat_preds = preds.reshape((preds.shape[0], -1))
    # Initialize a dummy "unlabeled" mask with the threshold
    flat_preds_with_treshold = torch.full(
        (preds.shape[0] + 1, flat_preds.shape[-1]), threshold
    )
    flat_preds_with_treshold[1 : preds.shape[0] + 1, :] = flat_preds

    # Get the top mask index for each pixel
    semantic_inds = torch.topk(flat_preds_with_treshold, 1, dim=0).indices.reshape(
        (preds.shape[-2], preds.shape[-1])
    )

    return semantic_inds


def clipseg_segmentation(
    processor, model, image, category_names, background_threshold, device
):
    inputs = processor(
        text=category_names,
        images=[image] * len(category_names),
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # resize the outputs
    logits = nn.functional.interpolate(
        outputs.logits.unsqueeze(1),
        size=(image.size[1], image.size[0]),
        mode="bilinear",
    )
    preds = torch.sigmoid(logits.squeeze())
    semantic_inds = preds_to_semantic_inds(preds, background_threshold)
    return preds, semantic_inds


def semantic_inds_to_shrunken_bool_masks(
    semantic_inds, shrink_kernel_size, num_categories
):
    shrink_kernel = np.ones((shrink_kernel_size, shrink_kernel_size))

    bool_masks = torch.zeros((num_categories, *semantic_inds.shape), dtype=bool)
    for category in range(num_categories):
        binary_mask = semantic_inds == category
        shrunken_binary_mask_array = (
            ndimage.binary_erosion(binary_mask.numpy(), structure=shrink_kernel)
            if shrink_kernel_size > 0
            else binary_mask.numpy()
        )
        bool_masks[category] = torch.from_numpy(shrunken_binary_mask_array)

    return bool_masks


def clip_and_shrink_preds(semantic_inds, preds, shrink_kernel_size, num_categories):
    # convert semantic_inds to shrunken bool masks
    bool_masks = semantic_inds_to_shrunken_bool_masks(
        semantic_inds, shrink_kernel_size, num_categories
    ).to(preds.device)

    sizes = [
        torch.sum(bool_masks[i].int()).item() for i in range(1, bool_masks.size(0))
    ]
    max_size = max(sizes)
    relative_sizes = [size / max_size for size in sizes]

    # use bool masks to clip preds
    clipped_preds = torch.zeros_like(preds)
    for i in range(1, bool_masks.size(0)):
        float_mask = bool_masks[i].float()
        clipped_preds[i - 1] = preds[i - 1] * float_mask

    return clipped_preds, relative_sizes


def sample_points_based_on_preds(preds, N):
    height, width = preds.shape
    weights = preds.ravel()
    indices = np.arange(height * width)

    # Randomly sample N indices based on the weights
    sampled_indices = random.choices(indices, weights=weights, k=N)

    # Convert the sampled indices into (col, row) coordinates
    sampled_points = [(index % width, index // width) for index in sampled_indices]

    return sampled_points


def upsample_pred(pred, image_source):
    pred = pred.unsqueeze(dim=0)
    original_height = image_source.shape[0]
    original_width = image_source.shape[1]

    larger_dim = max(original_height, original_width)
    aspect_ratio = original_height / original_width

    # upsample the tensor to the larger dimension
    upsampled_tensor = F.interpolate(
        pred, size=(larger_dim, larger_dim), mode="bilinear", align_corners=False
    )

    # remove the padding (at the end) to get the original image resolution
    if original_height > original_width:
        target_width = int(upsampled_tensor.shape[3] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :, :target_width]
    else:
        target_height = int(upsampled_tensor.shape[2] * aspect_ratio)
        upsampled_tensor = upsampled_tensor[:, :, :target_height, :]
    return upsampled_tensor.squeeze()


def sam_mask_from_points(predictor, image_array, points):
    points_array = np.array(points)
    # we only sample positive points, so labels are all 1
    points_labels = np.ones(len(points))
    # we don't use predict_torch here cause it didn't seem to work...
    _, _, logits = predictor.predict(
        point_coords=points_array,
        point_labels=points_labels,
    )
    # max over the 3 segmentation levels
    total_pred = torch.max(torch.sigmoid(torch.tensor(logits)), dim=0)[0].unsqueeze(
        dim=0
    )
    # logits are 256x256 -> upsample back to image shape
    upsampled_pred = upsample_pred(total_pred, image_array)
    return upsampled_pred


def inds_to_segments_format(
    panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
):
    panoptic_inds_array = panoptic_inds.numpy().astype(np.uint32)
    bitmap_file = bitmap2file(panoptic_inds_array, is_segmentation_bitmap=True)
    segmentation_bitmap = Image.open(bitmap_file)

    stuff_category_ids = [
        category_name_to_id[stuff_category_name]
        for stuff_category_name in stuff_category_names
    ]

    unique_inds = np.unique(panoptic_inds_array)
    stuff_annotations = [
        {"id": i, "category_id": stuff_category_ids[i - 1]}
        for i in range(1, len(stuff_category_names) + 1)
        if i in unique_inds
    ]
    thing_annotations = [
        {"id": len(stuff_category_names) + 1 + i, "category_id": thing_category_id}
        for i, thing_category_id in enumerate(thing_category_ids)
    ]
    annotations = stuff_annotations + thing_annotations

    return segmentation_bitmap, annotations


def generate_panoptic_mask(
    image,
    thing_category_names_string,
    stuff_category_names_string,
    dino_box_threshold=0.3,
    dino_text_threshold=0.25,
    segmentation_background_threshold=0.1,
    shrink_kernel_size=20,
    num_samples_factor=1000,
    task_attributes_json="",
):
    if task_attributes_json != "":
        task_attributes = json.loads(task_attributes_json)
        categories = task_attributes["categories"]
        category_name_to_id = {
            category["name"]: category["id"] for category in categories
        }
        # split the categories into "stuff" categories (regions w/o instances)
        # and "thing" categories (objects/instances)
        stuff_categories = [
            category
            for category in categories
            if "has_instances" not in category or not category["has_instances"]
        ]
        thing_categories = [
            category
            for category in categories
            if "has_instances" in category and category["has_instances"]
        ]
        stuff_category_names = [category["name"] for category in stuff_categories]
        thing_category_names = [category["name"] for category in thing_categories]
        category_names = thing_category_names + stuff_category_names
    else:
        # parse inputs
        thing_category_names = [
            thing_category_name.strip()
            for thing_category_name in thing_category_names_string.split(",")
        ]
        stuff_category_names = [
            stuff_category_name.strip()
            for stuff_category_name in stuff_category_names_string.split(",")
        ]
        category_names = thing_category_names + stuff_category_names
        category_name_to_id = {
            category_name: i for i, category_name in enumerate(category_names)
        }

    image = image.convert("RGB")
    image_array = np.asarray(image)

    # detect boxes for "thing" categories using Grounding DINO
    thing_boxes, thing_category_ids, detected_thing_category_names = dino_detection(
        dino_model,
        image,
        image_array,
        thing_category_names,
        category_name_to_id,
        dino_box_threshold,
        dino_text_threshold,
        device,
    )
    # compute SAM image embedding
    sam_predictor.set_image(image_array)
    # get segmentation masks for the thing boxes
    thing_masks = sam_masks_from_dino_boxes(
        sam_predictor, image_array, thing_boxes, device
    )
    # get rough segmentation masks for "stuff" categories using CLIPSeg
    clipseg_preds, clipseg_semantic_inds = clipseg_segmentation(
        clipseg_processor,
        clipseg_model,
        image,
        stuff_category_names,
        segmentation_background_threshold,
        device,
    )
    # remove things from stuff masks
    combined_things_mask = torch.any(thing_masks, dim=0)
    clipseg_semantic_inds_without_things = clipseg_semantic_inds.clone()
    clipseg_semantic_inds_without_things[combined_things_mask[0]] = 0
    # clip CLIPSeg preds based on non-overlapping semantic segmentation inds (+ optionally shrink the mask of each category)
    # also returns the relative size of each category
    clipsed_clipped_preds, relative_sizes = clip_and_shrink_preds(
        clipseg_semantic_inds_without_things,
        clipseg_preds,
        shrink_kernel_size,
        len(stuff_category_names) + 1,
    )
    # get finer segmentation masks for the "stuff" categories using SAM
    sam_preds = torch.zeros_like(clipsed_clipped_preds)
    for i in range(clipsed_clipped_preds.shape[0]):
        clipseg_pred = clipsed_clipped_preds[i]
        # for each "stuff" category, sample points in the rough segmentation mask
        num_samples = int(relative_sizes[i] * num_samples_factor)
        if num_samples == 0:
            continue
        points = sample_points_based_on_preds(clipseg_pred.cpu().numpy(), num_samples)
        if len(points) == 0:
            continue
        # use SAM to get mask for points
        pred = sam_mask_from_points(sam_predictor, image_array, points)
        sam_preds[i] = pred
    sam_semantic_inds = preds_to_semantic_inds(
        sam_preds, segmentation_background_threshold
    )
    # combine the thing inds and the stuff inds into panoptic inds
    panoptic_inds = sam_semantic_inds.clone()
    ind = len(stuff_category_names) + 1
    for thing_mask in thing_masks:
        # overlay thing mask on panoptic inds
        panoptic_inds[thing_mask.squeeze()] = ind
        ind += 1

    panoptic_bool_masks = (
        semantic_inds_to_shrunken_bool_masks(panoptic_inds, 0, ind + 1)
        .numpy()
        .astype(int)
    )
    panoptic_names = (
        ["unlabeled"] + stuff_category_names + detected_thing_category_names
    )
    subsection_label_pairs = [
        (panoptic_bool_masks[i], panoptic_name)
        for i, panoptic_name in enumerate(panoptic_names)
    ]

    segmentation_bitmap, annotations = inds_to_segments_format(
        panoptic_inds, thing_category_ids, stuff_category_names, category_name_to_id
    )
    annotations_json = json.dumps(annotations)

    return (image_array, subsection_label_pairs), segmentation_bitmap, annotations_json


config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename = "groundingdino_swint_ogc.pth"
sam_checkpoint = "./sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if device != "cpu":
    try:
        from GroundingDINO.groundingdino import _C
    except:
        warnings.warn(
            "Failed to load custom C++ ops. Running on CPU mode Only in groundingdino!"
        )

# initialize groundingdino model
dino_model = load_model_hf(config_file, ckpt_repo_id, ckpt_filename, device)

# initialize SAM
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

clipseg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
clipseg_model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined"
)
clipseg_model.to(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Panoptic Segment Anything demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    print(f"args = {args}")

    block = gr.Blocks(title="Panoptic Segment Anything").queue()
    with block:
        with gr.Column():
            title = gr.Markdown(
                "# [Panoptic Segment Anything](https://github.com/segments-ai/panoptic-segment-anything)"
            )
            description = gr.Markdown(
                "Demo for zero-shot panoptic segmentation using Segment Anything, Grounding DINO, and CLIPSeg."
            )
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source="upload", type="pil")
                    thing_category_names_string = gr.Textbox(
                        label="Thing categories (i.e. categories with instances), comma-separated",
                        placeholder="E.g. car, bus, person",
                    )
                    stuff_category_names_string = gr.Textbox(
                        label="Stuff categories (i.e. categories without instances), comma-separated",
                        placeholder="E.g. sky, road, buildings",
                    )
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        box_threshold = gr.Slider(
                            label="Grounding DINO box threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.001,
                        )
                        text_threshold = gr.Slider(
                            label="Grounding DINO text threshold",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.25,
                            step=0.001,
                        )
                        segmentation_background_threshold = gr.Slider(
                            label="Segmentation background threshold (under this threshold, a pixel is considered background/unlabeled)",
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.001,
                        )
                        shrink_kernel_size = gr.Slider(
                            label="Shrink kernel size (how much to shrink the mask before sampling points)",
                            minimum=0,
                            maximum=100,
                            value=20,
                            step=1,
                        )
                        num_samples_factor = gr.Slider(
                            label="Number of samples factor (how many points to sample in the largest category)",
                            minimum=0,
                            maximum=1000,
                            value=1000,
                            step=1,
                        )
                        task_attributes_json = gr.Textbox(
                            label="Task attributes JSON",
                        )

                with gr.Column():
                    annotated_image = gr.AnnotatedImage()
                    with gr.Accordion("Segmentation bitmap", open=False):
                        segmentation_bitmap_text = gr.Markdown(
                            """
The segmentation bitmap is a 32-bit RGBA png image which contains the segmentation masks.
The alpha channel is set to 255, and the remaining 24-bit values in the RGB channels correspond to the object ids in the annotations list.
Unlabeled regions have a value of 0.
Because of the large dynamic range, the segmentation bitmap appears black in the image viewer.
"""
                        )
                        segmentation_bitmap = gr.Image(
                            type="pil", label="Segmentation bitmap"
                        )
                        annotations_json = gr.Textbox(
                            label="Annotations JSON",
                        )

            examples = gr.Examples(
                examples=[
                    [
                        "a2d2.png",
                        "car, bus, person",
                        "road, sky, buildings, sidewalk",
                    ],
                    [
                        "bxl.png",
                        "car, tram, motorcycle, person",
                        "road, buildings, sky",
                    ],
                ],
                fn=generate_panoptic_mask,
                inputs=[
                    input_image,
                    thing_category_names_string,
                    stuff_category_names_string,
                ],
                outputs=[annotated_image, segmentation_bitmap, annotations_json],
                cache_examples=True,
            )

        run_button.click(
            fn=generate_panoptic_mask,
            inputs=[
                input_image,
                thing_category_names_string,
                stuff_category_names_string,
                box_threshold,
                text_threshold,
                segmentation_background_threshold,
                shrink_kernel_size,
                num_samples_factor,
                task_attributes_json,
            ],
            outputs=[annotated_image, segmentation_bitmap, annotations_json],
            api_name="segment",
        )

    block.launch(server_name="0.0.0.0", debug=args.debug, share=args.share)
