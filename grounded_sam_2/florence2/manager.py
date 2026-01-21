# -*- coding: utf-8 -*-
""" grounded_sam_2/florence2/manager.py """

from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from transformers.generation.utils import GenerateBeamEncoderDecoderOutput

from grounded_sam_2.florence2.task_prompts import FlorenceTasks
from grounded_sam_2.florence2.constants import BBOX_SCORES_LABEL


__all__ = [
    'Florence2MGR',
]


class Florence2MGR:
    """
    Contains methods to instantiate and manage the Florence2 model
    """

    def __init__(
            self,
            model_id: str = "florence-community/Florence-2-large",
            device: str | None = None, /, *,
            torch_dtype: str = 'auto'
    ):
        """
            model_id <str>: model_id for the
                            transformers.Florence2ForConditionalGeneration.from_pretrained method
                            Default "florence-community/Florence-2-large"
            device   <str>: torch device. If not provided, it is set automatically
        """
        assert isinstance(model_id, str), type(model_id)
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert isinstance(device, str), type(device)
        assert isinstance(torch_dtype, str), type(torch_dtype)

        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype

        # build florence-2
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype
        ).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def __call__(self, task_prompt: str, image: Image.Image, text_input: str | None = None) -> list:
        return self.run(task_prompt, image, text_input)

    def run(self, task_prompt: str, image: Image.Image, text_input: str | None = None) -> list:
        FlorenceTasks.validate(task_prompt)
        assert isinstance(image, Image.Image), type(image)
        if text_input is not None:
            assert isinstance(text_input, str), type(text_input)

        prompt = task_prompt if text_input is None else f'{task_prompt}{text_input}'
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].to(self.device),
            pixel_values=inputs["pixel_values"].to(self.device),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated_text = self.processor.batch_decode(
            generated_ids['sequences'], skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return generated_ids, generated_text, parsed_answer

    def compute_bboxes_confidence_scores(
            self,
            task_prompt: str,
            parsed_answer: dict,
            generated_ids: GenerateBeamEncoderDecoderOutput
    ):
        """
        Computes the bboxes confidence scores and adds them to the parsed_answer

        Inspired on: https://huggingface.co/microsoft/Florence-2-large/discussions/55
        """
        FlorenceTasks.validate(task_prompt)
        assert isinstance(parsed_answer, dict), type(parsed_answer)
        assert isinstance(generated_ids, GenerateBeamEncoderDecoderOutput), type(generated_ids)

        # computing transition scores
        transition_scores = self.model.compute_transition_scores(
            generated_ids.sequences,
            generated_ids.scores,
            generated_ids.beam_indices,
            normalize_logits=False,
        )
        # selecting relevant parts of arrays
        bounding_box_tokens = generated_ids.sequences[0][2:-1].cpu().numpy()
        bounding_box_scores = transition_scores[0][1:-1].cpu().numpy()

        # selecting values corresponding to bboxes
        bounding_box_indexes = np.where(
            np.logical_and(bounding_box_tokens >= 50269, bounding_box_tokens <= 51268)
        )
        bounding_box_scores = bounding_box_scores[bounding_box_indexes]

        # computing bboxes scores based on scores of their corresponding 4 points
        score_split_arrays = np.exp(
            np.mean(
                np.array_split(bounding_box_scores, len(bounding_box_scores) / 4),
                axis=1,
            )
        )

        # adding bboses confidence scores to the parsed_answer dictionary
        parsed_answer[task_prompt][BBOX_SCORES_LABEL] = score_split_arrays.tolist()

    @staticmethod
    def filter_bboxes_by_confidence(
            task_prompt: str, parsed_answer: dict, score: float, *, inplace: bool = False) -> dict[list]:
        """
        Filters thee bboxes using the provided score
        """
        FlorenceTasks.validate_bbox_task(task_prompt)
        assert isinstance(parsed_answer, dict), type(parsed_answer)
        assert isinstance(score, float), type(score)
        assert isinstance(inplace, bool), type(inplace)

        parsed_answer = parsed_answer if inplace else deepcopy(parsed_answer)
        data = parsed_answer[task_prompt]
        idxs = np.array(data[BBOX_SCORES_LABEL]) > score

        for key in FlorenceTasks.get_parsing_labels(task_prompt) + (BBOX_SCORES_LABEL, ):
            data[key] = np.array(data[key])[idxs].tolist()

        return parsed_answer

    @staticmethod
    def print_bbox_labels_scores(task_prompt: str, parsed_answer: dict):
        """
        Prints labels and scores from parsed_answer
        """
        FlorenceTasks.validate_bbox_task(task_prompt)
        assert isinstance(parsed_answer, dict), type(parsed_answer)

        bbox_labels_key = FlorenceTasks.get_parsing_labels(task_prompt)[0]

        for label, score in zip(parsed_answer[task_prompt][bbox_labels_key],
                                parsed_answer[task_prompt][BBOX_SCORES_LABEL]):
            print(f'{label}: {score}')
