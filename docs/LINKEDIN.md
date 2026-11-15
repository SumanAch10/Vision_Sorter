# LinkedIn post draft

I built **Vision Sorter**, a computer-vision application that combines recycling material classification with real semantic segmentation of litter.

The project now covers more than a model prediction:

• A fine-tuned ResNet-18 classifies six material categories.
• A U-Net with a pretrained ResNet encoder predicts litter/background pixels from human-annotated TACO masks.
• The interface overlays the mask and its centroid, shows alternative material predictions, and explains why a frame needs review.
• A FastAPI service provides typed responses, OpenAPI documentation, authentication, request limits, health checks, and metrics.

Measured results:

— **89.0% material classification accuracy** on 381 held-out TrashNet images.
— **0.51 foreground IoU / 0.61 Dice** on 313 held-out TACO images at 256px evaluation resolution.
— **28 passing local tests** covering geometry, model integrity, uncertainty handling, simulation gates, and API boundaries.

The two models are evaluated on separate datasets; these numbers are not a combined sorting success rate. Masks are binary litter/background, not per-pixel material labels.

One lesson from the project: a plausible mask and a confident class score still need independent checks. A silver-can regression remains misclassified, so the application flags it for review and issues no pick.

The release includes locked dependencies, model checksums, a Python wheel, and Docker/Compose configuration. Target-host container verification, cloud rollout, and physical robot validation are still ahead. The application currently operates in simulation only.

Built with Python, PyTorch, torchvision, FastAPI, and vanilla JavaScript. Thanks to the TrashNet and TACO dataset authors for making this work possible.

#ComputerVision #MachineLearning #MLOps #SemanticSegmentation

## Attachments

Use the images in `docs/assets/` (also generated under `artifacts/showcase/`):

1. `linkedin-project-card.png` — project overview and separate task metrics.
2. `linkedin-evaluation.png` — material F1 scores and distribution of segmentation IoU, including failures.
3. `linkedin-failure-analysis.png` — uncertainty and rejection on the silver-can regression.

These graphics contain measured data and diagrams, not source dataset photographs. The model-bundle archive is a developer artifact, not an attachment for LinkedIn. Add your own repository URL only after publishing the code; no public repository or live cloud URL has been created during this task.

The text is prepared for you to review and post. Nothing has been posted to LinkedIn.
