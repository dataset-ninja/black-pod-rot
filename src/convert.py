# https://www.kaggle.com/datasets/kenfackbruno/black-pod-rot-and-pod-borer-on-cocoa-pod

import os
import shutil
from collections import defaultdict
from urllib.parse import unquote, urlparse

import numpy as np
import supervisely as sly
from cv2 import connectedComponents
from dataset_tools.convert import unpack_if_archive
from dotenv import load_dotenv
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from supervisely.io.json import load_json_file
from tqdm import tqdm

import src.settings as s


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # project_name = "cocoa pod"
    dataset_path = "/mnt/d/datasetninja-raw/black-pod-rot"
    images_folder = "images"
    anns_ext = ".json"
    batch_size = 30
    download_bbox = True

    def create_ann(image_id):
        labels = []
        ann_data = image_id_to_ann_data[image_id]
        for label in ann_data:
            exterior = []
            for coords in label[1]:
                for i in range(0, len(coords), 2):
                    exterior.append([coords[i + 1], coords[i]])
            poligon = sly.Polygon(exterior)
            label_poly = sly.Label(poligon, category_id_to_classes[label[0]][0])
            labels.append(label_poly)
            if download_bbox:
                rectangle = sly.Rectangle(
                    top=label[2][1],
                    left=label[2][0],
                    bottom=label[2][1] + label[2][3],
                    right=label[2][0] + label[2][2],
                )
                label_rectangle = sly.Label(rectangle, category_id_to_classes[label[0]][1])
                labels.append(label_rectangle)

        return sly.Annotation(img_size=(image_id_to_shape[image_id]), labels=labels)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta()

    obj_classes_names = []
    category_id_to_classes = {}

    for ds_name in os.listdir(dataset_path):
        image_id_to_name = {}
        image_id_to_shape = {}
        image_id_to_ann_data = defaultdict(list)
        curr_ds_path = os.path.join(dataset_path, ds_name)

        images_path = os.path.join(curr_ds_path, images_folder)
        masks_json = os.path.join(curr_ds_path, "COCO_" + ds_name + anns_ext)
        images_names = os.listdir(images_path)

        json_ann = load_json_file(masks_json)

        categories = json_ann["categories"]
        for category in categories:
            if category["name"] not in obj_classes_names:
                obj_classes_names.append(category["name"])
                obj_class = sly.ObjClass(category["name"], sly.Polygon, color=category["color"])
                meta = meta.add_obj_class(obj_class)
                category_id_to_classes[category["id"]] = [obj_class]
                if download_bbox:
                    obj_class_bbox = sly.ObjClass(
                        category["name"] + "_bbox", sly.Rectangle, color=category["color"]
                    )
                    meta = meta.add_obj_class(obj_class_bbox)
                    category_id_to_classes[category["id"]].append(obj_class_bbox)
                api.project.update_meta(project.id, meta.to_json())

        images_data = json_ann["images"]
        for image_data in images_data:
            image_id_to_name[image_data["id"]] = image_data["file_name"]
            image_id_to_shape[image_data["id"]] = (image_data["height"], image_data["width"])
        annotations = json_ann["annotations"]
        for ann in annotations:
            segmentation_data = []
            for curr_coord in ann["segmentation"]:
                segmentation_data.append(list(map(int, curr_coord)))

            image_id_to_ann_data[ann["image_id"]].append(
                [ann["category_id"], segmentation_data, ann["bbox"]]
            )

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

        images_ids = list(image_id_to_name.keys())
        for images_ids_batch in sly.batched(images_ids, batch_size=batch_size):
            img_names_batch = [image_id_to_name[ids] for ids in images_ids_batch]
            img_pathes_batch = [
                os.path.join(images_path, image_id_to_name[ids]) for ids in images_ids_batch
            ]

            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_id) for image_id in images_ids_batch]
            api.annotation.upload_anns(img_ids, anns)

            progress.iters_done_report(len(img_names_batch))
    return project
