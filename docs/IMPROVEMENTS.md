# Historical classification upgrade (version 0.2.0)

The report below records the previous classification-only release. Version 0.3.0 adds real binary semantic segmentation and deployment hardening; see [README](../README.md), [release evidence](RELEASE_REPORT.json), and [deployment runbook](DEPLOYMENT.md) for current capabilities.

# Implemented improvements and measured results

## What changed

The default real-photo path now uses ResNet-18 pretrained on ImageNet and adapted to real TrashNet photographs. The original colored-shape U-Net is preserved as a separate, explicitly labeled demo. The application no longer presents its synthetic mask as evidence for real-photo material classification.

The material classifier covers cardboard, glass, metal, paper, plastic, and trash. Its training pipeline hashes image contents, removes exact duplicates, excludes contradictory duplicate labels, records stratified splits, trains a frozen feature head, and optionally fine-tunes the final residual stage. Validation loss selects checkpoints; validation NLL selects a temperature for probability calibration.

The interface now shows top-three alternatives, uncertainty, calibration-aware score labels, model identity, and held-out evaluation evidence. Image previews apply orientation consistently, oversized images are rejected before allocating large overlay canvases, and upload controls are disabled during inference to avoid displaying a response over a different image.

The API preserves the existing response fields and adds confidence semantics, material alternatives, score margin, and classification status. Classification-only decisions have no masks, centers, pick coordinates, or destination. Even changing a model's approval flag cannot enable picks in that path.

## Results from this run

Seed: 42. Original archive: 2,527 images. Three pairs of identical images had contradictory labels; all six files were excluded. Remaining images: 2,521, with 1,762 training, 378 validation, and 381 test images.

| Measurement | Result |
|---|---:|
| Synthetic baseline accuracy on 224 shared metal/plastic/paper test images | 31.7% |
| Frozen real-photo classifier on those same 224 images | 88.4% |
| Selected fine-tuned classifier on those same 224 images | 91.5% |
| Frozen classifier, all six materials, 381 test images | 83.7% |
| Selected classifier, all six materials, same 381 test images | 89.0% |
| Selected classifier macro F1 | 88.4% |
| Accuracy among 252 predictions passing score/margin checks | 97.2% |
| Fraction passing those score/margin checks | 66.1% |

The shared-class comparison maps the original `can` label to `metal` and `plastic_bottle` to `plastic`. It excludes cardboard and glass because the synthetic baseline had no corresponding labels. This comparison evaluates material classification only, not segmentation or robot performance.

For the selected model, validation-fitted temperature is approximately 1.7523. On the test set, temperature scaling changed NLL from 0.3985 to 0.3470, expected calibration error from 0.0520 to 0.0377, and Brier score from 0.1748 to 0.1725. It does not change the top class. Passing score checks does not authorize a robot pick.

The raw confusion matrix, per-class metrics, calibration measurements, and benchmark specification are in [evaluation.json](evaluation.json). Regenerate the comparison with `python scripts/evaluate_models.py` after training. Full training histories, image hashes, exclusions, and split memberships are in the local model manifest. The frozen checkpoint is retained locally at `artifacts/frozen-model`; the selected model is at `artifacts/real-model`.

## The user's silver-can photo remains a failure

`demo1.jpeg` was not added to training. The first frozen real-photo model predicted glass with a 64.8% score. The selected model's top alternatives are glass 49.2%, plastic 40.4%, and metal 8.7%. The UI flags this as uncertain because both the 85% score requirement and 15 percentage-point margin requirement fail. It issues no pick.

This is an improvement in measured dataset accuracy and in abstaining on this example, not a claim that this particular can is now classified correctly. Better coverage of plain silver cans, grayscale appearances, reflective surfaces, and external backgrounds is still needed. This photo should remain a regression example, not be repeatedly trained on and presented as independent proof of accuracy.

## Verification and limits

All 15 tests passed after model selection. They cover HTTP health/readiness, UI serving, API error responses, response shape, masks, coordinate transforms, checksums, preprocessing, calibration behavior, split isolation, contradictory labels, confidence ambiguity, and classification-only rejection. The updated JavaScript passed syntax checking. A live HTTP request confirmed that the station at port 8000 loads the selected real-photo model and rejects the silver-can photo as uncertain.

No browser visual inspection or Docker runtime verification was available in this session. No physical robot is connected. Real-data segmentation and localization still require annotated masks and independent calibration measurements.

The random test split was inspected while developing these changes. It is useful for this reported comparison but is not an untouched final external benchmark. Exact-content deduplication does not remove visually similar images or enforce capture-session isolation. A new external collection of labeled tabletop photos and representative unknown objects is the next basis for model selection and safety evaluation.
