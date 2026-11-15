# Understanding the code

## The complete version 0.3.0 path

The default product loads two independently trained models: a six-material classifier from `classification.py` and a binary litter/background segmenter from `litter_segmentation.py`. Both receive the validated image. The material model predicts one label for the frame; its label is not painted onto the segmenter's pixels.

`LitterUNet` reuses a pretrained ResNet-18 encoder. The encoder reduces spatial resolution while learning richer features. The decoder upsamples those features and concatenates matching encoder skip features to retain boundary detail. Two output logits per pixel represent background and litter.

`prepare_image` preserves aspect ratio by fitting the frame inside a 256×256 letterbox. `prepare_mask` applies the same geometry using nearest-neighbor interpolation and marks padding with 255. Cross-entropy and Dice loss ignore that padding. This matters: counting padded pixels as correct background would inflate scores.

At inference, `restore_probability` removes the letterbox padding before resizing the probability field to original image coordinates. Thresholding gives a binary semantic mask. `pipeline.py:_localize` calculates the mask's centroid and bounding box, checks area, foreground score, frame contact, and significant connected regions, and optionally projects the centroid with a validated calibration. Connected components are a heuristic; this is not learned instance segmentation.

The material classifier and the semantic segmenter are evaluated on separate datasets. Their scores are not multiplied into a fabricated joint accuracy or grasp confidence. A segmentation pixel score is uncalibrated, and a calibrated class score can still be wrong. Both model approvals remain false in the release, so no pick is authorized.

`schemas.py` defines the public response and cross-field invariants: masks must fit the image, runs cannot overlap or exceed bounds, image centers must be in bounds, and a review decision cannot contain an executed pick. `server.py` implements the FastAPI lifespan, bearer authentication, bounded image streaming, rate limiting, serialized inference, request IDs, sanitized model cards, and operational metrics. The browser uses `static/app.js` and `static/app.css`.

## Start with the real-photo path

1. `cli.py` parses `serve` or `infer` and chooses the model directory. A model is not merely Python architecture code: it also needs the learned weights and a manifest describing their meaning.
2. `pipeline.py:Sorter.__init__` validates architecture, class ordering, temperature, and the weight checksum. It creates the correct architecture and loads the saved parameters. Serving does not download or train a model.
3. `Sorter.infer` validates image bytes and dimensions, applies EXIF orientation, converts to RGB, and checks exposure/contrast. The model architecture determines the inference path.
4. `classification.py:photo_tensor` uses the official pretrained ResNet preprocessing: resize the shorter edge to 256, center crop to 224, tensor conversion, and ImageNet mean/std normalization. The tensor is CHW; a batch adds a leading dimension, producing `[N,3,224,224]`. Cropping is consistent with training but may remove objects near the edge.
5. ResNet produces `[N,6]` logits. A logit is an unnormalized score, not a probability. The model applies `softmax(logits / temperature)` to obtain six scores summing to one. The labels describe materials: cardboard, glass, metal, paper, plastic, trash.
6. `_classify` ranks the scores, records the top three, and checks the top score and its gap from the runner-up. Failing either check marks classification uncertain. Even passing those checks does not prove the label is correct or that the object is familiar.
7. The classifier alone has no spatial output. If no real segmenter is loaded, masks and coordinates are unavailable and every decision needs review. With a segmenter loaded, the complete path above adds a learned mask and centroid; millimeter coordinates still require calibration and all approval gates.
8. `server.py` returns the typed JSON-shaped response and logs request/model/decision metadata. `index.html` shows the scores, warnings, evidence, and original image. It never turns a material prediction into a fabricated mask.

## How learning changes the model

`classification.py:train_classifier` loads pretrained ResNet-18 features. The backbone converts each image into a 512-dimensional vector. The initial training stage freezes that backbone, computes vectors once, and learns a new six-output linear head.

For each training example, cross-entropy is `-log(probability assigned to the true class)`. Backpropagation computes how each trainable parameter changes that loss. AdamW updates those parameters. Validation loss selects a checkpoint; test examples never contribute gradients.

`fine_tune_classifier` then unfreezes `layer4` and the classification head. Image crops, flips, small rotations, and color changes provide varied views of the training images. Earlier residual stages stay frozen, including their batch-normalization statistics. The best validation checkpoint may still be the original frozen model if fine-tuning does not improve validation loss.

A seeded stratified split keeps each material represented in train, validation, and test. Content hashes detect exact duplicate files before the split. Contradictory duplicate labels are excluded rather than arbitrarily trusted. This does not detect near duplicates or prevent photos from the same capture session appearing in multiple sets.

## What confidence means

A class score describes the model's preference among its six known labels. It does not directly measure real-world correctness. A model can confidently force an unfamiliar object into one of those classes.

Temperature scaling fits one positive scalar to minimize validation cross-entropy. A temperature above one generally softens the distribution; below one sharpens it. This changes scores but not the top class. It cannot fix an incorrect ranking. It can improve validation NLL without improving every calibration metric on test data.

The report includes accuracy, per-class precision/recall/F1, confusion matrix, negative log-likelihood, expected calibration error, Brier score, and coverage. Coverage is the fraction passing score and margin checks. Accuracy among those examples is not pick safety: localization, workspace validity, and actuation remain separate questions.

## The separate segmentation path

`segmentation.py` implements a small U-Net. Its output is `[N,5,128,128]`: background plus four initial object classes at each pixel. `training.py` trains it against aligned integer masks using pixel cross-entropy. Nearest-neighbor resizing preserves discrete mask IDs.

The synthetic demo encodes classes using shape and color. Its strong synthetic scores do not establish real-photo performance. Its displayed score averages class probabilities over predicted foreground pixels, a different quantity from the real classifier's class score.

`pipeline.py` resizes the predicted mask back to the original image dimensions, selects the dominant foreground class, computes its pixel centroid, and applies mask quality checks. It rejects objects touching the frame, competing classes, fragmented regions, invalid area, and low confidence.

`calibration.py` fits a planar homography. Multiplying `[x,y,1]` by its matrix and dividing by the third coordinate gives tabletop coordinates. Independent reference measurements are needed to validate it. A low fitting residual alone is not proof of calibration accuracy.

`simulate` accepts only ready decisions. Demo models are never ready. Classification-only models are never ready. The combined path requires both real model approvals, mask gates, and calibration. A reviewed real segmentation model still needs every image, mask, calibration, workspace, and confidence gate to pass. No code here controls a physical motor.

## Files to read in order

- `data.py`: original object classes and image example contract.
- `classification.py`: real-photo preprocessing, training, calibration, and evaluation.
- `pipeline.py`: inference and decision gates.
- `litter_segmentation.py`: real binary litter model, paired geometry, training, and IoU/Dice evaluation.
- `schemas.py` and `settings.py`: API invariants and validated runtime configuration.
- `segmentation.py` and `training.py`: the separate educational pixel-label demo.
- `calibration.py`: pixel-to-table geometry.
- `cli.py`, `server.py`, `index.html`, and `static/`: product entry points.
- `tests/`: executable examples of accepted inputs, rejected inputs, and boundary conditions.

The original `model.py:TinySorterCNN` is only the foundation classification baseline used by `scripts/smoke_test.py`; it is not the deployed real-photo model.
