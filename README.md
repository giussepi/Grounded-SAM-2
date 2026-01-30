# Grounded SAM 2: Ground and Track Anything in Videos

This fork modified the code from  [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) to be easily installed and used as a third-party package. For now we re-wrote only the code from `grounded_sam_2/grounded_sam2_florence2_image_demo.py`


[[`SAM 2 Paper`](https://arxiv.org/abs/2408.00714)] [[`Grounding DINO Paper`](https://arxiv.org/abs/2303.05499)] [[`Grounding DINO 1.5 Paper`](https://arxiv.org/abs/2405.10300)] [[`DINO-X Paper`](https://arxiv.org/abs/2411.14347)] [[`BibTeX`](#citation)]

[![Video Name](./assets/grounded_sam_2_intro.jpg)](https://github.com/user-attachments/assets/f0fb0022-779a-49fb-8f46-3a18a8b4e893)

## Highlights

 Grounded SAM 2 is a foundation model pipeline towards grounding and track anything in Videos with [Grounding DINO](https://arxiv.org/abs/2303.05499), [Grounding DINO 1.5](https://arxiv.org/abs/2405.10300), [Florence-2](https://arxiv.org/abs/2311.06242), [DINO-X](https://arxiv.org/abs/2411.14347) and [SAM 2](https://arxiv.org/abs/2408.00714).

In this repo, we've supported the following demo with **simple implementations**:
- **Ground and Segment Anything** with Grounding DINO, Grounding DINO 1.5 & 1.6, DINO-X and SAM 2
- **Ground and Track Anything** with Grounding DINO, Grounding DINO 1.5 & 1.6, DINO-X and SAM 2
- **Detect, Segment and Track Visualization** based on the powerful [supervision](https://github.com/roboflow/supervision) library.

Grounded SAM 2 does not introduce significant methodological changes compared to [Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks](https://arxiv.org/abs/2401.14159). Both approaches leverage the capabilities of open-world models to address complex visual tasks. Consequently, we try to **simplify the code implementation** in this repository, aiming to enhance user convenience.

## Installation

1. Install CUDA>=12.1 and set the environment variable properly. E.g. `export CUDA_HOME=/path/to/cuda-12.1/`
2. Install PyTorch>=3.10 and torchvision>=0.18.1 with CUDA support following the [Pytorch installation instructions](https://pytorch.org/get-started/locally/).
3. Install it via PIP
   ```
   pip install git+https://github.com/giussepi/Grounded-SAM-2.git@packaged --no-cache-dir
   ```
   - If you get the following error: `AttributeError: install_layout. Did you mean: 'install_platlib'?`
	   * [Solution](https://github.com/lasp/cdflib/issues/167#issuecomment-1234019321)
       * run `export SETUPTOOLS_USE_DISTUTILS=stdlib` in a terminal
4. Open a python shell and download SAM 2 pretrained weights by importing the following modules:
   ```
   from grounded_sam_2 import download_pretrained_sam2
   # from grounded_sam_2 import download_pretrained_grounding_dino # not necessary for now
   ```
5. Using SAM 3 checkpoints requires access to its Hugging Face [repo](https://huggingface.co/facebook/sam3):
   1. Log in to you Hugging Face account, request access and wait for approval
   2. Authenticate yourself to access the checkpoints. Follow the instructions in [this page.](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)

## Uninstallation
1. Use PIP
   ```
   pip uninstall grounded_sam_2 -y
   ```
2. Get rid of the downloaded weights by removing its installation folder
   ```
   sudo rm -r <PATH TO YOUR PYTHON ENVIRONMENT>/lib/<YOUR PYTHON VERSION>/site-packages/grounded_sam_2/
   ```


## Usage
For now we re-wrote only the code from `grounded_sam_2/grounded_sam2_florence2_image_demo.py`. Thus, it can be used as follows:

```
from grounded_sam_2.grounded_sam2_florence2_image_demo import Sam2Florence2MGR

# IMAGE OPTION1 : using image path
image_path = "<image path>"

# IMAGE OPTION 2: grayscale PIL image
image_path = PIL.Image.fromarray(cv2.imread("<image_path>", 0))

# IMAGE OPTION 3: RGB PIL image (OpenCv.imread load images as BGR, so inverting axes is necessary)
#                 the model works better with RGB images than grayscale images
image_path = PIL.Image.fromarray(cv2.imread("<image_path>")[:, :, ::-1])


sm_mgr = Sam2Florence2MGR(
    sam2_enabled=True,
    sam3_enabled=True,
    # sam3_processor_kwargs= None,
    verbose=True,
    # print_fn=logger.info
)

sm_mgr.run_pipeline()
results, masks, masks_scores, masks_logits = sm_mgr.run_pipeline(
    image_path=image_path,
    pipeline="open_vocabulary_detection_segmentation",
    input_text="person <and> audience <and> football <and> advertisement <and> football pitch",
    bbox_conf_score_threshold=.4,  # None,
    atomic_query=True,
    plot_detections=True,
    return_values=True,
    use_sam3=True, # set it to False to use Florence2 + Sam2
)
sm_mgr.print_results_as_table(results, keys=['bboxes_labels', 'bboxes_scores', 'bboxes'], prepend_msg='OVD')


results, masks, masks_scores, masks_logits = sm_mgr.run_pipeline(
    image_path=image_path,
    pipeline="object_detection_segmentation",
    bbox_conf_score_threshold=None, # .2
	filter_labels=None, # ["person", "audience", "football"],
    plot_detections=True,
    return_values=True,
)
sm_mgr.print_results_as_table(results, prepend_msg='OD')

sm_mgr.run_pipeline(
    image_path=image_path,
    pipeline="region_proposal_segmentation"
)


# Using one line leveraging the __call__ method:
results, masks, masks_scores, masks_logits = Sam2Florence2MGR()(
    image_path=image_path,
    pipeline="open_vocabulary_detection_segmentation",
    input_text="person <and> audience <and> football <and> advertisement <and> football pitch",
    bbox_conf_score_threshold=.4,  # None,
    atomic_query=True,
    plot_detections=True,
    return_values=True,
    use_sam3=True,
)
sm_mgr.print_results_as_table(results, prepend_msg='OVD')
```

## TODO:
* Update `grounded_sam_2/grounded_sam2_florence2_image_demo.py->Sam2Florence2MGR`
  - [ ] Enable atomic queries in <CAPTION_TO_PHRASE_GROUNDING> and <REFERRING_EXPRESSION_SEGMENTATION>
  - [x] object_detection_and_segmentation
  - [ ] dense_region_caption_and_segmentation
  - [ ] region_proposal_and_segmentation
  - [ ] phrase_grounding_and_segmentation
  - [ ] referring_expression_segmentation
  - [x] open_vocabulary_detection_and_segmentation
