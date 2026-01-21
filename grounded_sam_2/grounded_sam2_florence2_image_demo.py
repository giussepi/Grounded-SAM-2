# -*- coding: utf-8 -*-
""" grounded_sam_2/grounded_sam2_florence2_image_demo.py """

import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image

from grounded_sam_2.florence2.manager import Florence2MGR
from grounded_sam_2.sam2.build_sam import build_sam2
from grounded_sam_2.sam2.sam2_image_predictor import SAM2ImagePredictor


__all__ = [
    'Sam2Florence2MGR',
]


# Define Some Hyperparam
TASK_PROMPT = {
    "caption": "<CAPTION>",
    "detailed_caption": "<DETAILED_CAPTION>",
    "more_detailed_caption": "<MORE_DETAILED_CAPTION",
    "object_detection": "<OD>",
    "dense_region_caption": "<DENSE_REGION_CAPTION>",
    "region_proposal": "<REGION_PROPOSAL>",
    "phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
    "region_to_segmentation": "<REGION_TO_SEGMENTATION>",
    "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    "region_to_category": "<REGION_TO_CATEGORY>",
    "region_to_description": "<REGION_TO_DESCRIPTION>",
    "ocr": "<OCR>",
    "ocr_with_region": "<OCR_WITH_REGION>",
}

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Sam2Florence2MGR:
    """
    Holds methos to run different pipelines using sam2 and Florence2

    Usage:
        # Using manager instance
        sm_mgr = Sam2Florence2MGR()
        sm_mgr.run_pipeline()
        sm_mgr.run_pipeline(
            image_path="<path to image>",
            pipeline="open_vocabulary_detection_segmentation",
            input_text="person <and> crowd <and> football",
            verbose=True,
            plot_detections=True,
        )

        # USING CALL METHOD
        Sam2Florence2MGR()(
            image_path="<path to image>",
            pipeline="open_vocabulary_detection_segmentation",
            input_text="person <and> crowd <and> football",
            verbose=True,
            plot_detections=True,
        )
    """

    def __init__(self, *,
                 output_dir: str = "./outputs",
                 florence2_model_id: str = "florence-community/Florence-2-large",
                 sam2_checkpoint: str = "sam2.1_hiera_large.pt",
                 sam2_config: str = "sam2.1_hiera_l.yaml",
                 ):
        """
        Initializes the Sam2Florence2MGR manager.

        Kwargs:
            output_dir <str>: output directory path
                              Default "./outputs"
            florence2_model_id <str>: model id for the
                              transformers.Florence2ForConditionalGeneration.from_pretrained method
                              Default "florence-community/Florence-2-large"
            sam2_checkpoint <str>: one of the sam2 checkpoint located at
                              "grounded_sam_2/checkpoints/".
                              Default "sam2.1_hiera_large.pt"
            sam2_config <str>: one of the YAML config files located at
                              "grounded_sam_2/sam2/configs/sam2.1/"
                              Default "sam2.1_hiera_l.yaml"
        """
        assert isinstance(output_dir, str), type(output_dir)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        assert isinstance(florence2_model_id, str), type(florence2_model_id)
        sam2_checkpoint = os.path.join(
            CURRENT_DIR, "checkpoints", sam2_checkpoint
        )
        assert Path(sam2_checkpoint).is_file(), sam2_checkpoint
        sam2_config = os.path.join("configs", "sam2.1", sam2_config)
        assert (Path(CURRENT_DIR)/"sam2"/sam2_config).is_file(), sam2_config

        self.florence2_model_id = florence2_model_id
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.florence2_mgr, self.sam2_model, self.sam2_predictor = self.build_models()

    def __call__(self, *,
                 image_path: str = "./grounded_sam_2/notebooks/images/cars.jpg",
                 pipeline: str = "object_detection_segmentation",
                 input_text: str | None = None,
                 **kwargs):
        """
        Excecutes the specified pipeline over the provided image

        Kwargs:
            image_path <str>: path to image to be processed.
                              Default: ./grounded_sam_2/notebooks/images/cars.jpg
            pipeline   <str>: pipeline name to be executed.
                              Default: object_detection_segmentation
            input_text <str>: pipeline input text.
                              Default None

        """
        return self.run_pipeline(image_path=image_path, pipeline=pipeline, input_text=input_text, **kwargs)

    @staticmethod
    def get_image(image_path: str | Image.Image) -> Image.Image:
        """ Returns an Image.Image instance in RGB mode """
        assert isinstance(image_path, (str, Image.Image)), type(image_path)

        image = Image.open(image_path) if isinstance(image_path, str) else image_path

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def build_models(self) -> tuple:
        """
        Returns:
            florence2_model, florence2_processor, sam2_model, sam2_predictor
        """
        # build florence-2
        florence2_mgr = Florence2MGR(self.florence2_model_id, self.device)

        # build sam 2
        sam2_model = build_sam2(
            self.sam2_config, self.sam2_checkpoint, device=self.device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        return florence2_mgr, sam2_model, sam2_predictor

    def run_florence2(self, task_prompt, text_input, image) -> list:
        results = self.florence2_mgr(task_prompt, image, text_input)

        return results

    """
    We support a set of pipelines built by Florence-2 + SAM 2
    """

    """
    Pipeline-1: Object Detection + Segmentation
    """

    def object_detection_and_segmentation(
        self,
        image_path: str | Image.Image,
        bbox_conf_score_threshold: float | None = None,
        verbose: bool = True,
        plot_detections: bool = True,
        return_values: bool = False,
    ) -> tuple:
        """
        Kwargs:
            image_path <str | Image.Image>: PIL.Image instance (RGB or grayscale) or path to image to be
                              processed. NOTE: It works better with RGB images.
            bbox_conf_score_threshold <float | None>: bbox confidence score threshold. E.g. 0.5
                              Default None
            verbose   <bool>: Whether or not print extra messages.
                              Default True
            plot_detections <bool>: Whether or plot and save detections to disk.
                              Default True
            return_values <bool>: Whether or not return values.
                              Default False

        Returns:
            tuple(
                results         <dict | None>: dictionary containing 'bboxes', 'labels'
                masks        <ndarray | None>: binary ndarray [N, H, W]
                masks scores <ndarray | None>: ndarray [N, 1]
                masks logits <ndarray | None>: ndarray [N, 1, 256, 256]
            )
        """
        if bbox_conf_score_threshold is not None:
            assert isinstance(bbox_conf_score_threshold, float), type(bbox_conf_score_threshold)
        assert isinstance(return_values, bool), type(return_values)

        # NOTE: text_input must be None when calling object detection pipeline.
        text_input = None
        task_prompt = "<OD>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Object Detection Output Format
        {'<OD>':
            {
                'bboxes':
                    [
                        [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469],
                        [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906],
                        [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438],
                        [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406],
                        [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                    ],
                'labels': ['car', 'door', 'door', 'wheel', 'wheel']
            }
        }
        """
        # TODO: refactor the following lines
        if bbox_conf_score_threshold is not None:
            self.florence2_mgr.compute_bboxes_confidence_scores(task_prompt, results, generated_ids)
            # self.florence2_mgr.print_bbox_labels_scores(task_prompt, results) # For debugging
            self.florence2_mgr.filter_bboxes_by_confidence(
                task_prompt, results, bbox_conf_score_threshold, inplace=True)
            # self.florence2_mgr.print_bbox_labels_scores(task_prompt, results) # For debugging
            # returning if there are no results after applying the confidence threshold
            if len(results[task_prompt]['bboxes']) == 0:
                if verbose:
                    print(f"No detections were found after filfering results using "
                          f"bbox the confidence score: {bbox_conf_score_threshold}")
                if return_values:
                    return None, None, None, None
                return

        results = results[task_prompt]
        # parse florence-2 detection results
        input_boxes = np.array(results["bboxes"])
        if verbose:
            print(results)
        class_names = results["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if plot_detections:
            # specify labels
            labels = [
                f"{class_name}" for class_name in class_names
            ]

            # visualization results
            img = np.array(image)
            img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids
            )

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_det_annotated_image.jpg"), annotated_frame)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_det_image_with_mask.jpg"), annotated_frame)

            if verbose:
                print(f'Successfully save annotated image to "{self.output_dir}"')

        if return_values:
            return results, masks, scores, logits

    """
    Pipeline 2: Dense Region Caption + Segmentation
    """

    def dense_region_caption_and_segmentation(
        self,
        image_path: str | Image.Image,
        text_input=None,
    ):
        assert text_input is None, "Text input should be None when calling dense region caption pipeline."
        task_prompt = "<DENSE_REGION_CAPTION>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Object Detection Output Format
        {'<DENSE_REGION_CAPTION>':
            {
                'bboxes':
                    [
                        [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469],
                        [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906],
                        [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438],
                        [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406],
                        [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                    ],
                'labels': ['turquoise Volkswagen Beetle', 'wooden double doors with metal handles', 'wheel', 'wheel', 'door']
            }
        }
        """
        results = results[task_prompt]
        # parse florence-2 detection results
        input_boxes = np.array(results["bboxes"])
        class_names = results["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # specify labels
        labels = [
            f"{class_name}" for class_name in class_names
        ]

        # visualization results
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_dense_region_cap_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_dense_region_cap_image_with_mask.jpg"), annotated_frame)

        print(f'Successfully save annotated image to "{self.output_dir}"')

    """
    Pipeline 3: Region Proposal + Segmentation
    """

    def region_proposal_and_segmentation(
        self,
        image_path: str | Image.Image,
        text_input=None,
    ):
        assert text_input is None, "Text input should be None when calling region proposal pipeline."
        task_prompt = "<REGION_PROPOSAL>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Object Detection Output Format
        {'<REGION_PROPOSAL>':
            {
                'bboxes':
                    [
                        [33.599998474121094, 159.59999084472656, 596.7999877929688, 371.7599792480469],
                        [454.0799865722656, 96.23999786376953, 580.7999877929688, 261.8399963378906],
                        [224.95999145507812, 86.15999603271484, 333.7599792480469, 164.39999389648438],
                        [449.5999755859375, 276.239990234375, 554.5599975585938, 370.3199768066406],
                        [91.19999694824219, 280.0799865722656, 198.0800018310547, 370.3199768066406]
                    ],
                'labels': ['', '', '', '', '', '', '']
            }
        }
        """
        results = results[task_prompt]
        # parse florence-2 detection results
        input_boxes = np.array(results["bboxes"])
        class_names = results["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # specify labels
        labels = [
            f"region_{idx}" for idx, class_name in enumerate(class_names)
        ]

        # visualization results
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_region_proposal.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_region_proposal_with_mask.jpg"), annotated_frame)

        print(f'Successfully save annotated image to "{self.output_dir}"')

    """
    Pipeline 4: Phrase Grounding + Segmentation
    """

    def phrase_grounding_and_segmentation(
        self,
        image_path: str | Image.Image,
        text_input=None,
    ):
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Object Detection Output Format
        {'<CAPTION_TO_PHRASE_GROUNDING>':
            {
                'bboxes':
                    [
                        [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594],
                        [1.5999999046325684, 4.079999923706055, 639.0399780273438, 305.03997802734375]
                    ],
                'labels': ['A green car', 'a yellow building']
            }
        }
        """
        assert text_input is not None, "Text input should not be None when calling phrase grounding pipeline."
        results = results[task_prompt]
        # parse florence-2 detection results
        input_boxes = np.array(results["bboxes"])
        class_names = results["labels"]
        class_ids = np.array(list(range(len(class_names))))

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # specify labels
        labels = [
            f"{class_name}" for class_name in class_names
        ]

        # visualization results
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_phrase_grounding.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_phrase_grounding_with_mask.jpg"), annotated_frame)

        print(f'Successfully save annotated image to "{self.output_dir}"')

    """
    Pipeline 5: Referring Expression Segmentation

    Note that Florence-2 directly support referring segmentation with polygon output format, which may be not that accurate,
    therefore we try to decode box from polygon and use SAM 2 for mask prediction
    """

    def referring_expression_segmentation(
        self,
        image_path: str | Image.Image,
        text_input=None,
    ):
        task_prompt = "<REFERRING_EXPRESSION_SEGMENTATION>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Object Detection Output Format
        {'<REFERRING_EXPRESSION_SEGMENTATION>':
            {
                'polygons': [[[...]]]
                'labels': ['']
            }
        }
        """
        assert text_input is not None, "Text input should not be None when calling referring segmentation pipeline."
        results = results[task_prompt]
        # parse florence-2 detection results
        polygon_points = np.array(results["polygons"][0], dtype=np.int32).reshape(-1, 2)
        class_names = [text_input]
        class_ids = np.array(list(range(len(class_names))))

        # parse polygon format to mask
        img_width, img_height = image.size[0], image.size[1]
        florence2_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        if len(polygon_points) < 3:
            print("Invalid polygon:", polygon_points)
            exit()
        cv2.fillPoly(florence2_mask, [polygon_points], 1)
        if florence2_mask.ndim == 2:
            florence2_mask = florence2_mask[None]

        # compute bounding box based on polygon points
        x_min = np.min(polygon_points[:, 0])
        y_min = np.min(polygon_points[:, 1])
        x_max = np.max(polygon_points[:, 0])
        y_max = np.max(polygon_points[:, 1])

        input_boxes = np.array([[x_min, y_min, x_max, y_max]])

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        sam2_masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if sam2_masks.ndim == 4:
            sam2_masks = sam2_masks.squeeze(1)

        # specify labels
        labels = [
            f"{class_name}" for class_name in class_names
        ]

        # visualization florence2 mask
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=florence2_mask.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(self.output_dir, "florence2_referring_segmentation_box.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "florence2_referring_segmentation_box_with_mask.jpg"), annotated_frame)

        print(f'Successfully save florence-2 annotated image to "{self.output_dir}"')

        # visualize sam2 mask
        img = np.array(image)
        img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=sam2_masks.astype(bool),
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_referring_box.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_referring_box_with_sam2_mask.jpg"), annotated_frame)

        print(f'Successfully save sam2 annotated image to "{self.output_dir}"')

    """
    Pipeline 6: Open-Vocabulary Detection + Segmentation
    """

    def open_vocabulary_detection_and_segmentation(
        self,
        image_path: str | Image.Image,
        text_input,
        bbox_conf_score_threshold: float | None = None,
        verbose: bool = True,
        plot_detections: bool = True,
        return_values: bool = False,
    ) -> tuple:
        """
        Kwargs:
            image_path <str | Image.Image>: PIL.Image instance (RGB or grayscale) or path to image to be
                              processed. NOTE: It works better with RGB images.
            text_input <str>: object to be found. Several objects can be specified
                              using <and> separator. E.g. "person <and> crowd <and>
                              football"
            bbox_conf_score_threshold <float | None>: bbox confidence score threshold. E.g. 0.5
                              Default None
            verbose   <bool>: Whether or not print extra messages.
                              Default True
            plot_detections <bool>: Whether or plot and save detections to disk.
                              Default True
            return_values <bool>: Whether or not return values.
                              Default False

        Returns:
            tuple(
                results         <dict | None>: dictionary containing 'bboxes', 'bboxes_labels',
                                'polygons', 'polygons_labels'
                masks        <ndarray | None>: binary ndarray [N, H, W]
                masks scores <ndarray | None>: ndarray [N, 1]
                masks logits <ndarray | None>: ndarray [N, 1, 256, 256]
            )
        """
        assert text_input is not None, "Text input should not be None when calling open-vocabulary detection pipeline."
        if bbox_conf_score_threshold is not None:
            assert isinstance(bbox_conf_score_threshold, float), type(bbox_conf_score_threshold)
        assert isinstance(return_values, bool), type(return_values)

        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        # run florence-2 object detection in demo
        image = self.get_image(image_path)
        generated_ids, _, results = self.run_florence2(task_prompt, text_input, image)

        """ Florence-2 Open-Vocabulary Detection Output Format
        {'<OPEN_VOCABULARY_DETECTION>':
            {
                'bboxes':
                    [
                        [34.23999786376953, 159.1199951171875, 582.0800170898438, 374.6399841308594]
                    ],
                'bboxes_labels': ['A green car'],
                'polygons': [],
                'polygons_labels': []
            }
        }
        """
        if bbox_conf_score_threshold is not None:
            self.florence2_mgr.compute_bboxes_confidence_scores(task_prompt, results, generated_ids)
            # self.florence2_mgr.print_bbox_labels_scores(task_prompt, results) # For debugging
            self.florence2_mgr.filter_bboxes_by_confidence(
                task_prompt, results, bbox_conf_score_threshold, inplace=True)
            # self.florence2_mgr.print_bbox_labels_scores(task_prompt, results) # For debugging
            # returning if there are no results after applying the confidence threshold
            if len(results[task_prompt]['bboxes']) == 0:
                if verbose:
                    print(f"No bbox detections were found after filfering results using "
                          f"the confidence score: {bbox_conf_score_threshold}")
                if return_values:
                    return None, None, None, None
                return

        results = results[task_prompt]
        # parse florence-2 detection results
        input_boxes = np.array(results["bboxes"])
        if verbose:
            print(results)
        class_names = results["bboxes_labels"]
        class_ids = np.array(list(range(len(class_names))))

        # predict mask with SAM 2
        self.sam2_predictor.set_image(np.array(image))
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        if plot_detections:
            # specify labels
            labels = [
                f"{class_name}" for class_name in class_names
            ]

            # visualization results
            img = np.array(image)
            img = img[:, :, ::-1]  # RGB to BGR, required to use cv2.imwrite correctly
            detections = sv.Detections(
                xyxy=input_boxes,
                mask=masks.astype(bool),
                class_id=class_ids
            )

            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(self.output_dir, "grounded_sam2_florence2_open_vocabulary_detection.jpg"), annotated_frame)

            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.output_dir,
                        "grounded_sam2_florence2_open_vocabulary_detection_with_mask.jpg"), annotated_frame)

            if verbose:
                print(f'Successfully save annotated image to "{self.output_dir}"')

        if return_values:
            return results, masks, scores, logits

    def run_pipeline(self, *,
                     image_path: Image.Image | str = "",
                     pipeline: str = "object_detection_segmentation",
                     input_text: str | None = None,
                     **kwargs):
        """
        Excecutes the specified pipeline over the provided image

        Kwargs:
            image_path <str | Image.image>: PIL.Image instance (RGB or grayscale) or path to image to be
                              processed. NOTE: It works better with RGB images.
                              Default: './notebooks/images/cars.jpg'
            pipeline   <str>: pipeline name to be executed.
                              Default: 'object_detection_segmentation'
            input_text <str>: pipeline input text.
                              Default None
        """
        assert isinstance(image_path, (str, Image.Image)), type(image_path)
        if isinstance(image_path, str):
            image_path = image_path if image_path else os.path.join(
                CURRENT_DIR, "notebooks", "images", "cars.jpg"
            )
            assert Path(image_path).is_file(), image_path
        assert isinstance(pipeline, str), type(pipeline)
        if input_text is not None:
            assert isinstance(input_text, str), type(input_text)
        kwargs['verbose'] = kwargs.get('verbose', True)
        assert isinstance(kwargs['verbose'], bool), type(kwargs['verbose'])

        if kwargs['verbose']:
            print(f"Running pipeline: {pipeline} now.")

        match pipeline:
            case "object_detection_segmentation":
                # pipeline-1: detection + segmentation
                return self.object_detection_and_segmentation(
                    image_path=image_path,
                    **kwargs
                )
            case "dense_region_caption_segmentation":
                # pipeline-2: dense region caption + segmentation
                return self.dense_region_caption_and_segmentation(
                    image_path=image_path
                )
            case "region_proposal_segmentation":
                # pipeline-3: dense region caption + segmentation
                return self.region_proposal_and_segmentation(
                    image_path=image_path
                )
            case "phrase_grounding_segmentation":
                # pipeline-4: phrase grounding + segmentation
                return self.phrase_grounding_and_segmentation(
                    image_path=image_path,
                    text_input=input_text
                )
            case "referring_expression_segmentation":
                # pipeline-5: referring segmentation + segmentation
                return self.referring_expression_segmentation(
                    image_path=image_path,
                    text_input=input_text
                )
            case "open_vocabulary_detection_segmentation":
                # pipeline-6: open-vocabulary detection + segmentation
                return self.open_vocabulary_detection_and_segmentation(
                    image_path=image_path,
                    text_input=input_text,
                    **kwargs
                )
            case _:
                raise NotImplementedError(
                    f"Pipeline: {pipeline} is not implemented at this time")
