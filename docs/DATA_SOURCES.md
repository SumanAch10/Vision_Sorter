# Real-photo model sources

- Dataset: Gary Thung and Mindy Yang, [TrashNet](https://github.com/garythung/trashnet).
- Original image archive: https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip
- Archive SHA256: `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`.
- The author's MIT license is preserved in `TrashNet-LICENSE.txt` and alongside local images.
- Modification: duplicate byte-identical files are deduplicated; conflicting labels for identical content are excluded and recorded in the model manifest. The remaining images are stratified into seeded 70/15/15 train/validation/test splits. Original images are not modified on disk.
- ResNet-18 ImageNet weights: official [torchvision checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth), hash prefix `f37072fd`.
- [Official preprocessing documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): resize shorter edge to 256, center crop to 224, ImageNet normalization.

TrashNet labels are material categories, not verified object types. They do not support claiming that every metal object is a can or every plastic object is a bottle. Dataset test accuracy does not establish performance on arbitrary internet photos, cluttered scenes, or unknown materials.

## Real binary semantic segmentation

- Dataset: Pedro F. Proença and Pedro Simões, [TACO: Trash Annotations in Context](https://github.com/pedropro/TACO), [paper](https://arxiv.org/abs/2003.06975).
- Pinned annotation snapshot: commit `3844bcd3535cd12ec0218494d1d79f1372f553cc`, `data/annotations.json`.
- Snapshot SHA256: `812c2e3f0d336b437f276597715de2abb08bb8a641599c45f593d233ec66457b`.
- Images: author-hosted `https://img.tacodataset.org/thumb/` URLs from that snapshot. Other source hosts are excluded before sampling.
- Of 1,034 eligible thumbnails, two byte-identical copies (source IDs 1173 and 1305) are excluded. Remaining: 1,032. Source-batch-disjoint split: 622 train, 97 validation, 313 test. These batch labels are not a guarantee of independent acquisition sessions.
- All annotated litter polygons are scaled to actual thumbnail dimensions and unioned into binary masks: background=0, litter=1. Padding=255 is ignored during learning/evaluation. This does not provide material-specific semantic labels or instance IDs.
- Annotation transformations, source URLs, source license metadata, image/mask hashes, exclusions, and source groups are recorded in the processed dataset index and trained model manifest.
- The repository's MIT license is retained in `TACO-LICENSE.txt`. This is not a blanket assertion that every source photograph has the same license: some image license fields are unspecified. Source photos are kept local and are not included in model bundles or LinkedIn graphics. Check individual image rights before redistributing photographs.
- Both the classifier and segmenter start from official ImageNet-pretrained ResNet-18 weights. Their task heads, preprocessing, datasets, and evaluations are separate.
