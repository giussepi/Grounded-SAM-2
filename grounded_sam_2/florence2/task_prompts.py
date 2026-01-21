# -*- coding: utf-8 -*-
""" grounded_sam_2/florence2/task_prompts.py """

from grounded_sam_2.florence2.constants import BBOXES_LABEL


__all__ = [
    'FlorenceTasks',
]


class FlorenceTasks:
    """
    Holds the supported Florence2 tasks and some useful methods
    """
    OD = '<OD>'
    OVD = '<OPEN_VOCABULARY_DETECTION>'
    DRC = '<DENSE_REGION_CAPTION>'
    RP = '<REGION_PROPOSAL>'
    C2PG = '<CAPTION_TO_PHRASE_GROUNDING>'
    RES = '<REFERRING_EXPRESSION_SEGMENTATION>'

    OPTIONS = (OD, OVD, DRC, RP, C2PG, RES)
    LABELS = {
        OD: ('labels', BBOXES_LABEL),
        OVD: ('bboxes_labels', BBOXES_LABEL, 'polygons_labels', 'polygons'),
        DRC: ('labels', BBOXES_LABEL),
        RP: ('labels', BBOXES_LABEL),
        C2PG: ('labels', BBOXES_LABEL),
        RES: ('labels', 'polygons'),
    }

    @staticmethod
    def clean_option(option: str) -> str:
        """ returns the option ready for matching """
        return option.strip()

    @classmethod
    def validate(cls, option: str) -> str:
        """
        validates the provided option is among the class defined options

        Args:
            option <str>: one of FlorenceTasks.OPTIONS

        Returns:
            cleaned_option <str>
        """
        cleaned_option = cls.clean_option(option)
        assert cleaned_option in cls.OPTIONS, f'{cleaned_option} is not in {cls.__name__}.OPTIONS '

        return cleaned_option

    @classmethod
    def get_parsing_labels(cls, option: str) -> tuple:
        cleaned_option = cls.validate(option)

        return cls.LABELS[cleaned_option]
