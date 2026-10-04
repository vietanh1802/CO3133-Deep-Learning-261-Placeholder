# Assignment 2 - M1 Dataset Proposal

> Complete this proposal for one task track before beginning the main Assignment 2 implementation. Submit it for instructor approval. Do not treat this template as an approved dataset proposal.

## 1. Group and proposed task

- **Group name / ID:** Placeholder
- **Members:**

  | Full name        | Student ID |
  | ---------------- | ---------- |
  | Vu Duc Viet Anh  | 2352074    |
  | Tran Lam Anh     | 2352067    |
  | Le Nguyen Khang  | 2352470    |
  | Tran Nguyen Giap | 2352284    |

- **Task track**: Semantic Segmentation
- **Problem statement:** Assign every pixel of an underwater RGB image to one of eight categories, so that an autonomous underwater vehicle can separate divers, robots, reefs, wrecks, plants, fish, sea floor and open water in a single pass.
- **Prediction unit:** One pixel. An H by W image produces H times W predictions, not one label per image.
- **Input:** One RGB underwater photograph. The data holds 11 resolutions, from 416x416 to 1906x1080, with 640x480 on 83.7 percent of images.
- **Output / labels:** Eight mutually exclusive classes, encoded by the authors as 3-bit RGB colours (paper, Table I): BW background waterbody (000), HD human divers (001), PF plants and sea-grass (010), WR wrecks or ruins (011), RO robots and instruments (100), RI reefs and invertebrates (101), FV fish and vertebrates (110), SR sand, sea-floor and rocks (111). The bit pattern read as binary is the class index.
- **Why this task and dataset are suitable:** 1,540 usable masks and seven foreground classes clear the handbook thresholds of 1,000 and three. Supervision is per-pixel ground truth from the dataset authors, and an official 110-image test set keeps our numbers comparable with the published benchmark. The task is not trivial: classes are strongly imbalanced, and underwater colour attenuation and backscatter are absent from terrestrial pretraining, so the pretrained-versus-baseline comparison measures something real.

## 2. Dataset identity and access

- **Dataset name:** SUIM, Segmentation of Underwater IMagery. Islam et al., "Semantic Segmentation of Underwater Imagery: Dataset and Benchmark", IROS 2020, arXiv:2004.01241.
- **Source and download URL:** http://irvlab.cs.umn.edu/resources/suim-dataset.
- **Version / release / access date:** None published. Neither the paper nor the archive's `INFO.txt` carries a version. Downloaded 02 Oct 2026.
- **License and permitted use:** MIT, "Copyright (c) 2020 Md Jahidul Islam", from the `LICENSE` file at https://github.com/xahidbuffon/SUIM.
- **Data format and storage size:** JPEG images, 24-bit BMP masks, one mask per image with the same stem. Download 4.9 GB, of which this project uses 2.2 GB: `train_val/` 1.8 GB (1,525 pairs) and `TEST/` 428 MB (110 pairs). `Checkpoint_Data/` (2.8 GB) holds the authors' weights and is unused. `TEST/masks/` also holds eight subdirectories of single-channel binary masks: one per foreground class, BW excluded, plus a saliency mask.
- **Collection and annotation process, if known:** Images "carefully chosen from a large pool of samples collected during oceanic explorations and human-robot cooperative experiments in several locations of various water types", plus "a few images from large-scale datasets named EUVP, USR-248 and UFO-120", three earlier datasets by the same group. "All images of the SUIM dataset are pixel-annotated by seven human participants", following published guidelines for the confusable pairs plants versus reefs and vertebrates versus invertebrates.
- **Known limitations or biases:** (a) Part of the data is reused from EUVP, USR-248 and UFO-120, so the sample is not independent. (b) Classes are skewed: by the paper's Figure 2a, RO appears in 101 of 1,525 images and PF in 239, against RI 1,028 and FV 1,030. (c) Eleven resolutions and seven aspect ratios, so any fixed-size pipeline resamples a non-uniform source.

## 3. Dataset size and preliminary analysis

- **Total samples:** 1,635 image and mask pairs: 1,525 in `train_val/` and 110 in `TEST/`. Counted from disk by matching filename stems between the `images/` and `masks/` folder of each split. Every image has a mask and every mask has an image, so nothing is unmatched. The totals agree with `INFO.txt` and with the paper.
- **Proposed usable samples:** 1,540 pairs: 1,430 of the 1,525 in `train_val/` plus all 110 in `TEST/`. Every file decodes. Two rules remove the other 95, all from `train_val`. First, 85 masks were saved through JPEG, which shifted their colours; since SUIM stores a pixel's class as its colour, those pixels now read as the wrong class, and the error is systematic rather than scattered, with reef edges reading as robot. In a shifted mask 97 percent of robot pixels sit against a reef pixel, against 0 percent in a clean one, and robot is the rarest class and the one section 5 compares models on. Those 85 also include every pair whose mask is taller than its image. Second, 10 images are byte-level or near-pixel copies of a `TEST` image, so training on them would inflate the test score. `TEST` is left whole, to stay comparable with the published benchmark.
- **Annotation types and counts:** One dense pixel mask per image: 1,540 masks for 1,540 images, 559,013,360 labelled pixels in total. Every pixel carries one of the eight class codes and the dataset defines no void or ignore label, so nothing is left unlabelled and no pixel is excluded from the loss. No bounding boxes, polygons or image-level tags are shipped. `TEST/` adds 880 single-channel binary masks, one per image for each of the seven foreground classes plus a saliency mask, which are the authors' own per-class benchmark targets.
- **Class / target distribution:** three counts per class, because each answers a different question: how many images contain the class, what share of all pixels it owns, and how much of an image it covers on the occasions it does appear.

  | Class                        | `train_val` images | pixel share | share of the image when present | `TEST` images | pixel share |
  | ---------------------------- | -------------------- | ----------- | ------------------------------- | --------------- | ----------- |
  | BW background waterbody      | 1,185 (82.9%)        | 30.99%      | 33.6%                           | 95 (86.4%)      | 41.81%      |
  | HD human divers              | 359 (25.1%)          | 2.34%       | 4.9%                            | 42 (38.2%)      | 3.69%       |
  | PF plants and sea-grass      | 218 (15.2%)          | 2.29%       | 7.4%                            | 20 (18.2%)      | 3.08%       |
  | WR wrecks or ruins           | 245 (17.1%)          | 6.77%       | 43.2%                           | 27 (24.5%)      | 7.28%       |
  | RO robots and instruments    | 86 (6.0%)            | 0.56%       | 3.8%                            | 12 (10.9%)      | 0.90%       |
  | RI reefs and invertebrates   | 964 (67.4%)          | 35.64%      | 56.1%                           | 58 (52.7%)      | 19.47%      |
  | FV fish and vertebrates      | 958 (67.0%)          | 7.26%       | 7.2%                            | 66 (60.0%)      | 6.77%       |
  | SR sand, sea-floor and rocks | 571 (39.9%)          | 14.15%      | 29.5%                           | 60 (54.5%)      | 17.00%      |

  The classes are very unevenly sized. Reefs own 35 percent of every pixel in the data and robots own 0.56 percent, a gap of 63 times. Counted by images instead, the same gap is 11 times. The two counts do not rank the classes alike, so a plan built on either one alone would be built on the wrong order. Being rare also has two different causes here. Wrecks appear in only 245 images but fill 43 percent of
  each image they are in, while fish appear in 958 images and fill 7 percent. A model loses wrecks by missing whole scenes and loses fish by missing small objects, which are different failures.

  `TEST` is not a smaller copy of `train_val`: reefs fall from 36 to 19 percent of pixels and sea floor rises from 14 to 17. Per-class scores on the two splits are therefore not interchangeable. Before the 95 exclusions, these image counts matched the paper's Figure 2a within 2 percent for every class except robots, whose excess was the mislabelled fringe described above.
- **Input size or length distribution:** 11 distinct resolutions across the 1,540 pairs. 640x480 covers 83.8 percent and 960x540 another 8.3; the other nine share the remaining 7.9. Widths run from 416 to 1906 and heights from 360 to 1080, so the largest image holds 12 times the pixels of the smallest. There are seven aspect ratios between 1.000 and 1.778, five square images and no portrait one, and 84.7 percent of the data is 4:3. The paper lists 1906x1080, 1280x720, 640x480 and 256x256 as examples, but 256x256 does not occur at all and 960x540, the second most common, is not mentioned. `train_val` and `TEST` have nearly the same profile, 83.8 against 83.6 percent at 640x480, so resolution does not shift between the splits even though the class mix does. Every model will therefore see a resized input, and the resize rule has to be recorded and held identical across models for the comparison to mean anything.
- **Missing, invalid, duplicate, or low-quality records:**

  | Fault                                                | Count                | How it was found                                                                   |
  | ---------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------- |
  | Image with no mask, or mask with no image            | 0                    | filename stems compared between the two folders                                    |
  | File that fails to decode                            | 0                    | every image and mask fully decoded once                                            |
  | Mask taller than its image by 55 rows                | 37                   | sizes compared pair by pair                                                        |
  | Mask whose colours drifted off the eight class codes | 85                   | every mask pixel tested against the eight exact colours; the 37 above are a subset |
  | Duplicate image inside`train_val`                  | 115, in 55 groups    | file hash, then perceptual hash confirmed against pixels                           |
  | Duplicate image across`train_val` and `TEST`     | 10 groups, 20 images | the same procedure                                                                 |

  Duplicates were found in two steps. A file hash caught 13 groups of identical files. A perceptual hash then caught the same scene saved at a different resolution, but it flags far too much on images this flat and blue, so every pair it raised was checked against the real pixels: 2,035 flagged, 76 confirmed. A one-off sweep over 49,399 looser candidates found nothing further, so the routine search is wide enough.

  Ten of the 110 official `TEST` images, 9.1 percent, also sit in `train_val`, four of them identical byte for byte. That duplication came with the dataset. We leave `TEST` whole so our numbers stay comparable with the published benchmark, and drop the `train_val` copy instead.

  Dropping the 85 drifted masks costs real data. Of the 51 that claimed a robot, about 35 were the mislabelled fringe and about 16 held a real robot. So the rule throws away roughly 16 real robot images to remove 35 false ones, which is expensive for the rarest class, but a wrong label is worse than a missing one when section 5 scores models on that class.
- **Evidence (tables, plots, or analysis links):** `scripts/eda/run_suim.py` produces every number in this section in one run and writes it to `results/a2/eda/suim/`. The script measures nothing itself: it calls `src/data/suim.py` and applies the two exclusion rules, so the tables above and the files below cannot drift apart.

  | File                             | What it carries                                                                                                                 |
  | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
  | `summary.json`                 | every count above, the per-class figures for both splits, the resolution histogram, and each duplicate group listed by filename |
  | `class_distribution.png`       | image count beside pixel share, which is the disagreement described above                                                       |
  | `region_area_distribution.png` | the spread of area covered where a class is present, which separates rare-and-large from common-and-small                       |
  | `resolution_distribution.png`  | the eleven resolutions on a log scale                                                                                           |

  Because `summary.json` names the duplicate groups, the ten `TEST` images that also sit in `train_val` can be opened and checked one by one rather than taken on trust. The wider recall sweep is not part of a normal run; it was done once by raising `HASH_DISTANCE` from 12 to 20, and is repeated the same way.
  The dataset is not in the repository, so the script expects it unpacked at `data/suim/`.
- **Relevant task-specific checks:** five checks a classification dataset would never need.

  1. Image and mask must agree on size, or the labels do not line up with the pixels they label. 37 pairs failed and are excluded.
  2. Every mask pixel must sit on one of the eight class colours, since the colour *is* the label. 85 masks failed and are excluded.
  3. There is no void or ignore label, so every pixel counts in both the loss and the metric, and no`ignore_index` may be carried over from code written for Cityscapes or ADE20K.
  4. Classes per image run from 2 to 6 of the 8, with a median of 3. No image holds a single class and none holds all eight, so every image is a real multi-class segmentation problem. 245 of the 1,430 `train_val` images contain no open water at all.
  5. Class co-occurrence. Reefs share an image with 79.8 percent of the plant images and 77.9 percent of the fish images, the two highest rates in the data, which is why section 5 singles out those two pairs for the confusion matrix. Robots are the opposite: they share an image with wrecks 4 times and with plants 5 times in 1,430 images.
- **How the dataset meets the handbook's threshold for this track (or justification for an exception):**
  Handbook section 19.4 sets four conditions for segmentation, and no exception is needed.

  | Condition                                          | Status                                                                       |
  | -------------------------------------------------- | ---------------------------------------------------------------------------- |
  | At least 3 foreground classes, background excluded | 7, and all 7 appear in both splits                                           |
  | About 1,000 or more images with masks              | 1,540 usable pairs, of which 1,430 are available for training and validation |
  | Binary datasets only if approved                   | not applicable; the task is 8-way, not figure against ground                 |
  | Analyse per-class pixel distribution               | done above, counted by pixel share, by image and by area when present        |

  The margin over the 1,000 threshold survives the exclusions: 1,430 remain after removing 85 masks with drifted colours and 10 images that duplicate a `TEST` image.

## 4. Split and leakage prevention

- **Official or custom train / validation / test split:** partly official. The dataset ships `train_val/` and `TEST/` and no validation set. `TEST/` is used exactly as published, so our numbers stay comparable with the paper's benchmark. Train and validation are cut from `train_val/` by us, 80/20, by the procedure below.
- **Expected sample counts per split:**

  | Split | Pairs | Where it comes from |
  | --- | --- | --- |
  | train | 1,143 | folds 1 to 4 of `train_val` |
  | validation | 287 | fold 0 of `train_val` |
  | test | 110 | `TEST/` as published, untouched |
  | total | 1,540 | the usable pairs of section 3 |

  The ratio is 79.9 / 20.1 rather than exactly 80 / 20 because whole groups move together and groups are not all the same size.
- **Split unit (for example, subject, scene, document, or source video):** the group of confirmed duplicate images, not the single image. The 1,430 usable `train_val` images form 1,370 groups: 1,315 images that are unique, plus 55 groups that hold 115 images between them (51 pairs, 3 triples, 1 group of four). Filenames cannot serve as the unit. There are only four prefixes and `f_r` alone covers 929 of the 1,430 images, so grouping by prefix would put 65 percent of the data on one side.
- **Why this unit prevents leakage:** if a group were split across the boundary, validation would be scoring an image whose near-copy the model had already trained on, and validation mIoU would read high for the wrong reason. This is measured, not hypothetical: section 3 confirmed 76 duplicate pairs against the pixels. With the single image as the unit, a 20 percent draw separates a group of size *k* with probability 1 minus 0.8^*k* minus 0.2^*k*, so about 18 of the 55 groups would be expected to straddle the boundary. Keeping groups whole drives that to zero by construction.
- **Stratification or grouping policy:** each image is labelled with the rarest class it contains, where rarity is counted on these 1,430 images rather than taken from the paper. One label per image, and the scarce classes are what decide it. The fold is then stratified on that label while groups stay whole. Stratum sizes: FV 452, SR 264, WR 217, PF 213, HD 151, RO 86, RI 47. BW never names a stratum, because every image containing open water also contains something rarer.

  The reason for stratifying at all is RO, in 86 of 1,430 images. An unconstrained draw can leave validation with a handful of them, and section 5 scores rare-class IoU separately, so that number has to mean something. What the policy achieved:

  | Class | train images | val images | val share |
  | --- | --- | --- | --- |
  | BW | 949 | 236 | 19.9 % |
  | HD | 287 | 72 | 20.1 % |
  | PF | 175 | 43 | 19.7 % |
  | WR | 197 | 48 | 19.6 % |
  | RO | 69 | 17 | 19.8 % |
  | RI | 761 | 203 | 21.1 % |
  | FV | 760 | 198 | 20.7 % |
  | SR | 458 | 113 | 19.8 % |

  Every class lands within 1.1 points of 20 percent, and all eight appear in both halves.
- **Random seed and reproducible split procedure:** seed 42, `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` from scikit-learn, fold 0 as validation. `scripts/eda/run_suim.py` writes the result to `results/a2/splits/suim/train.txt` and `val.txt`, one filename stem per line, sorted, LF endings, and both files are committed to the repository. Their sha256 begins `5341b993d7f8d87d` and `0e59e81b28a0548e`; a re-run reproduces those bytes, which was checked. Training reads the two files instead of re-deriving the split, so a later edit to the code cannot quietly move an image from one side to the other.
- **Duplicate / near-duplicate checks across splits:** the procedure and its counts are in section 3 and are not repeated here. What matters for the split: 65 duplicate groups were confirmed, 10 of them crossing `train_val` and `TEST`. In all 10 the `train_val` member was dropped, so no training or validation image has a confirmed copy in the test set. The other 55 groups lie wholly inside `train_val` and are held together by the split unit above. The check runs over images, not masks, so a pair that was re-annotated rather than re-saved is still caught.
- **Test-set isolation plan:** `TEST/` is read once, at the end, for the one configuration of each model already selected on validation. Nothing else touches it: no tuning, no early stopping, no checkpoint selection, no threshold. All 110 pairs pass both exclusion rules of section 3, so none is discarded and the set is used as published, which `INFO.txt` asks for ("do not alter"). Test predictions are written once and not iterated on.
- **Subset selection rules, if any:** none. No image is subsampled or repeated: all 1,143 train, 287 validation and 110 test pairs are used in full, every epoch. The only images removed are those section 3 already accounts for, the 85 pairs that fail an exclusion rule and the 10 `train_val` duplicates of a `TEST` image. Class imbalance is not addressed by resampling, so any weighting is a model-side choice and belongs in section 6.

## 5. Evaluation plan

- **Primary metric(s) and why they fit the task:** Mean IoU and Dice over the eight classes, as required by handbook section 20.1. Both are computed by accumulating TP, FP and FN across the whole split and taking the per-class value from those totals, so each class counts once regardless of how many pixels it owns. That matters here: the paper's Figure 2a puts RO in 101 of 1,525 images against RI in 1,028, so pixel accuracy would stay high for a model that predicts only BW and SR, while mIoU exposes it. Dice is reported beside IoU because the gap between them widens on small regions, and showing only Dice would overstate performance on exactly the rare classes this task cares about.
- **Secondary / per-class or per-group metric(s):** Per-class IoU for all eight classes, with RO, PF and WR treated as the rare group. The authors ship one binary mask per class for the test split, so per-class scoring is how the dataset itself is benchmarked. Per-image mIoU, averaged over only the classes present in that image, is used solely to rank images for the qualitative selection below and is never reported as a model score.
- **Qualitative examples to inspect:** Fixed now, before any result exists. For each model: the 5 validation images with the highest per-image mIoU and the 5 with the lowest, plus 3 fixed images containing RO drawn with seed 42. Each is shown as input, ground truth and prediction. The lowest 5 are shown whatever they turn out to contain.
- **Error-analysis plan:** Three analyses, fixed in advance. (1) An 8 by 8 pixel-level confusion matrix, to identify which class pairs are exchanged; our own co-occurrence counts make PF against RI and FV against RI the likely pairs, since reefs share an image with 79.8 percent of plant images and 77.9 percent of fish images. (2) IoU computed separately on a narrow band around the ground-truth boundaries and on the region interiors, which separates a loose outline from a missed object. (3) Per-class IoU against the class's image count and its median region area, to test whether rare-class failure follows from rarity or from small region size.
- **Decision criterion for comparing models:** Model A beats model B only if its validation mIoU exceeds B's by more than 1.0 point AND it reduces no rare-class IoU (RO, PF, WR) by more than 1.0 point. A difference inside those bounds is reported as a tie, not as a win. Every comparison is made on validation; the test split is run once at the end, for the selected configuration of each model.

## 6. Model and experiment plan

Handbook section 20 asks for eight things. Items 5 to 8, quantitative and qualitative evaluation and error analysis, are section 5 of this proposal; the compute-cost analysis is section 7.

- **Simple baseline:** two, because they answer different questions.

  1. A constant predictor that labels every pixel RI, the class that owns the most pixels. It learns nothing and takes no time. RI covers 37.9 percent of validation pixels and the other seven classes score zero, so it scores mIoU 0.047. Its job is to prove the metric is wired correctly and to turn "better than the trivial answer" into a number rather than an assumption.
  2. A U-Net trained from scratch, base width 32, 7.8 M parameters, no pretrained weights. This is the baseline the comparison in section 5 is actually against.
- **Planned pretrained or modern model(s) and source/checkpoint version:** `torchvision.models.segmentation.deeplabv3_resnet50` with `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1` (torchvision 0.29.0, 160.5 MB, 66.4 mIoU on COCO-val2017 with VOC labels). The classifier head is replaced by a 1x1 convolution to 8 channels and the auxiliary head is dropped, which leaves 39.6 M parameters. If 4 GB of video memory proves too small, the replacement is `lraspp_mobilenet_v3_large` with `LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1`, 3.2 M parameters; section 7 explains when that switch is made.
- **Planned fine-tuning procedure:**

  | Setting | Value | Why |
  | --- | --- | --- |
  | Input size | 320x240 | 4:3, the exact aspect ratio of 83.7 percent of the images, so the common case is not distorted |
  | Resampling | bilinear for the image, nearest for the mask | nearest invents no new colour, and in this dataset a new colour is a new class |
  | Normalisation | ImageNet mean and standard deviation | the statistics the COCO checkpoint was trained with |
  | Augmentation | random horizontal flip only | no vertical flip, since underwater scenes have a fixed up direction; no colour jitter, since colour separates water from reef here |
  | Loss | cross-entropy over all 8 classes, no `ignore_index` | section 3, check 3: this dataset has no void label |
  | Optimiser | AdamW, head 1e-3, backbone 1e-4, weight decay 1e-4 | a pretrained backbone needs smaller steps than a fresh head |
  | Schedule | cosine, 5 epochs of linear warm-up, 40 epochs total | fits the compute budget in section 7 |
  | Batch | 4, with gradient accumulation of 4, effective 16 | 4 is what the memory allows; accumulation restores a usable batch |

  The U-Net baseline runs the same recipe with one learning rate of 1e-3, since it has no pretrained part. Everything else is held identical, so the baseline and the modern model differ only in architecture and pretraining.
- **Controlled experiment or ablation:** loss A against loss B, the "Loss A vs. B" option in handbook section 20.
  - **Hypothesis:** adding a Dice term to cross-entropy raises IoU on the rare classes (RO, PF, WR) by more than 1.0 point, and costs no more than 1.0 point of overall mIoU. The reason is measured, not borrowed: RO owns 0.52 percent of training pixels. Cross-entropy averages over pixels, so it is nearly indifferent to a class that small. Dice is computed per class and normalised by region size, so a missed robot costs it the same as a missed reef.
  - **Changed factor:** the loss function, and nothing else. Arm A is cross-entropy. Arm B is 0.5 cross-entropy plus 0.5 soft Dice, Dice averaged over the 8 classes.
  - **Fixed factors:** DeepLabV3-ResNet50 with the COCO checkpoint, 320x240 input, horizontal flip only, AdamW with the learning rates and schedule above, 40 epochs, effective batch 16, seed 42, and the same `train.txt` and `val.txt` committed in section 4.
  - **Metric(s) and decision criterion:** per-class IoU on RO, PF and WR, plus overall validation mIoU. Arm B is adopted only if at least one rare class gains more than 1.0 point of IoU and overall mIoU falls by no more than 1.0 point. A result inside that band is reported as no effect, not as a small win, and both arms are then re-run with seeds 1 and 2 before anything is written down, because a gap that small cannot be read off one run.

## 7. Compute and reproducibility estimate

- **Available hardware (device and memory):** one laptop, NVIDIA GeForce RTX 2050 with 4,096 MiB of video memory, driver 532.10, Windows 11, 8 CPU threads. The 4 GB is the binding constraint on every choice in section 6, and it is also shared with the display, so the usable figure is lower. One problem is open and must be fixed before M2: the environment currently holds `torch 2.14.0+cpu`, a CPU-only build, because `pyproject.toml` asks only for `torch>=2.2,<3` and the resolver took the default wheel. The CUDA build has to be installed and the lock file regenerated.
- **Estimated preprocessing, training, and evaluation time:** there is no offline preprocessing pass. Images are decoded and resized inside the data loader, and the one-off analysis of sections 3 and 4 runs end to end in under five minutes.

  Training cost was measured on this machine rather than guessed, at 320x240 and batch 4, which is 286 steps per epoch over the 1,143 training images:

  | Model | Parameters | CPU, per step | CPU, per epoch | CPU inference |
  | --- | --- | --- | --- | --- |
  | U-Net base 32 | 7.8 M | 8.56 s | 40.8 min | 449 ms/image |
  | DeepLabV3-ResNet50 | 39.6 M | 17.64 s | 84.1 min | 1,473 ms/image |
  | LR-ASPP MobileNetV3-Large | 3.2 M | 1.76 s | 8.4 min | 87 ms/image |

  These are rough, taken over a handful of steps on a busy machine, but they settle the question they were asked: 40 epochs of DeepLabV3 on the CPU is 56 hours, so the CPU is not a slow option, it is not an option. On the GPU, assuming the 20x to 30x speed-up usual for this class of card, an epoch should cost 2 to 4 minutes and a 40-epoch run 1 to 3 hours. That figure is a projection and is replaced by a measurement in M2. The three planned runs, U-Net and the two ablation arms, come to roughly 6 GPU hours; the conditional re-run on seeds 1 and 2 would add about 9 more. Evaluation is negligible: 287 validation images once per epoch, 110 test images once in total.
- **Estimated storage and checkpoint size:** under 1 GB of new files, next to the 2.2 GB of SUIM already on disk. Only the best epoch's weights are kept per run, 29.6 MB for U-Net and 151.2 MB for DeepLabV3, so 332 MB for the three runs. A resume checkpoint carrying AdamW state is 453.6 MB for DeepLabV3 and exists only while a run is in progress. The COCO checkpoint adds a 160.5 MB download to the torch cache, and predictions and figures stay under 100 MB. `checkpoints/` is git-ignored; only the split files, `summary.json` and the figures are committed.
- **Expected software framework and versions:** Python 3.12.7, PyTorch 2.14.0 (CUDA build required, see above), torchvision 0.29.0, NumPy 2.5.3, scikit-learn 1.9.1, Pillow 12.3.0, Matplotlib 3.11.2. Pinned in `pyproject.toml` and `uv.lock`, installed with `uv sync`. Mixed precision through `torch.amp`.
- **Planned seed(s) and checkpoint selection rule:** seed 42 for every reported run, set for Python, NumPy and PyTorch. The split itself is not re-randomised at all: training reads `train.txt` and `val.txt` from section 4, so the seed affects initialisation, shuffling and augmentation only. Seeds 1 and 2 are added to both ablation arms if their gap falls inside the 1.0 point band of section 6, since a result that small cannot be read off one run.

  The checkpoint kept is the one with the highest validation mIoU, evaluated after every epoch, ties going to the earlier epoch. This differs on purpose from A1, which kept the lowest validation loss. Cross-entropy here is dominated by BW and RI, which own 66 percent of the pixels between them, so the lowest-loss epoch is not reliably the best epoch for the rare classes that section 5 scores separately.
- **Feasibility risks and fallback within the same approved dataset/task:**

  | Risk | Fallback, in order |
  | --- | --- |
  | DeepLabV3 does not fit in 4 GB | batch 4 to 2 with accumulation raised to 8, so the effective batch stays 16; then input 320x240 to 256x192; then swap to LR-ASPP MobileNetV3-Large, which is a tenth of the parameters and was measured ten times faster per step |
  | The CUDA build cannot be made to work on this card | run the same script on a free Colab or Kaggle T4, which has 16 GB, against the same committed split files, so the numbers stay comparable |
  | 40 epochs overruns the M2 deadline of 28 Oct 2026 | cut to 25 epochs with the cosine schedule rescaled, applied equally to both ablation arms so the comparison stays fair |

  Every fallback stays on SUIM and on 8-class segmentation, so none of them needs a new approval.

## 8. Approval record

- **Submission date and location:** due 07 Oct 2026, 23:59 GMT+7. Submitted as `reports/a2_proposal.md` in this repository. <!-- TODO: confirm the exact submission channel on the course page before sending. -->
- **Instructor decision:** pending.
- **Decision date:** pending.
- **Conditions or requested changes:** none recorded.
- **Changes made in response:** none recorded.
