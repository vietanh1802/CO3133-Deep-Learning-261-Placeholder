# Assignment 3 - M1 Dataset Proposal

> Complete this proposal for one task track before beginning the main Assignment 3 implementation. Submit it for instructor approval. Do not treat this template as an approved dataset proposal.

## 1. Group and proposed task

- **Group name / ID:**
- **Members:**
- **Task track:**
- **Problem statement:**
- **Prediction unit:**
- **Input:**
- **Output / labels:**
- **Why this task and dataset are suitable:**

## 2. Dataset identity and access

- **Dataset name:**
- **Source and download URL:**
- **Version / release / access date:**
- **License and permitted use:**
- **Data format and storage size:** All 8,091 images are RGB JPEG files and occupy 1,114,733,885 bytes (about 1.1 GB). Captions, split lists and human judgements are tab-separated text files totalling 10,113,299 bytes (about 10 MB).
- **Collection and annotation process, if known:**
- **Known limitations or biases:**

## 3. Dataset size and preliminary analysis

- **Total samples:** The release contains 8,091 images. `Flickr8k.token.txt` yields 40,455 captions for these images, exactly five per image. Five further lines use the key `2258277193_586949ec62.jpg.1`, which names no image in the archive, so we drop them. Every image has captions and every caption names an existing image, so no modality is missing.
- **Proposed usable samples:** We use the 8,000 images listed in the official train, dev and test files, which give 40,000 image-caption pairs. The 91 images that appear in no split file are left out, so that our test set matches the published one. Every image decodes fully. The single confirmed duplicate image pair lies inside the training split and is kept, because it cannot leak into validation or test.
- **Annotation types and counts:** Each image has five independently written English captions, and `Flickr8k.lemma.token.txt` is a lemmatized copy of them. The release also contains human judgements of image-caption pairs, all on the 1,000 test images. `ExpertAnnotations.txt` scores 5,822 pairs from 1 (does not describe the image) to 4 (describes the image) by three experts. `CrowdFlowerAnnotations.txt` records yes and no votes, at least three per pair, for 47,830 pairs. We use the judgements only to assess pair quality and the evaluation protocol, never for training or model selection.
- **Class / target distribution:** Retrieval has no class labels: the target of an image is its own captions, and the target of a caption is its image. Caption vocabulary shows the content instead. In the 30,000 training captions, the most frequent content words are "dog" (6,159 occurrences) and "man" (5,374), followed by colour words and words for people (Appendix A). Scenes of dogs and people therefore dominate, and many test images will have visually and verbally similar neighbours. The training vocabulary has 7,692 words, 3,188 of which occur only once. Words absent from the training captions make up 1.34 percent of dev tokens and 1.22 percent of test tokens.
- **Input size or length distribution:** Image width ranges from 164 to 500 pixels (median 500) and height from 127 to 500 pixels (median 375), across 597 distinct sizes. Only 48 images have a short side below 224 pixels, the usual input size of pretrained image encoders. Training captions have 2 to 36 tokens, with median 10 and mean 10.79. The dev and test means are 10.90 and 10.84, and the length distributions of the three splits nearly coincide (Appendix A). A token is a lowercased word that contains at least one letter or digit.
- **Missing, invalid, duplicate, or low-quality records:** No image is missing captions, no caption is missing its image, and no image fails to decode. Apart from the five captions with a malformed key, the audit found:
  - **Duplicate images:** the perceptual-hash search returned 6,701 candidate pairs. A pixel comparison confirmed one, `2851198725_37b6027625.jpg` and `3050606344_af711c726c.jpg`, which have a hash distance of 0 and a pixel difference of 0.0 and are both in the training split.
  - **Very short captions:** 141 training, 19 dev and 28 test captions have at most three tokens. They are kept, because short captions are still valid queries.
  - **Mismatched pairs:** of the 721 test pairs in which CrowdFlower judged a caption against its own image, 700 (97.1 percent) received a majority of yes votes. The other 21 are listed in `summary.json` and will be inspected during error analysis. The experts judged 158 own pairs; the mean of the three scores rounds to 4 for 155 and to 3 for the remaining 3, and never lower.
- **Evidence (tables, plots, or analysis links):** `scripts/eda/run_flickr8k.py` writes every number in this section to `results/a3/eda/flickr8k/summary.json`, together with the figures in Appendix A.
- **Relevant task-specific checks:** Retrieval metrics count only the captions written for an image as correct. Two checks estimate how often this rule marks a correct result as wrong.
  - **Captions shared by several images:** in the test split, 7 caption texts are identical, after tokenization, for two or more images, involving 13 images in total. An example is "a surfer is riding a wave". Such a query has more than one correct image, but the metric accepts only one. The dev split has 5 such captions and the training split 142.
  - **Captions that also fit another image:** the experts judged 5,664 test pairs in which the caption was written for a different image. All three experts gave the highest score to 103 of them. CrowdFlower gave a majority of yes votes to 2,004 of its 47,109 such pairs. Recall@K will therefore slightly underestimate true retrieval quality. We will report the standard protocol unchanged and discuss these cases in the error analysis.
- **How the dataset meets the handbook's threshold for this track (or justification for an exception):** The handbook asks for about 5,000 or more image-text pairs for image-text retrieval. The training split alone provides 30,000 pairs from 6,000 images, and the usable dataset provides 40,000. Every caption was written for the image it is paired with, so the pairs are genuinely aligned rather than drawn from two independent datasets.

## 4. Split and leakage prevention

- **Official or custom train / validation / test split:** We use the official split files `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt` and `Flickr_8k.testImages.txt`, with the dev split as validation. No image is listed in more than one file, and no file lists an image twice. `summary.json` records the SHA-256 digest of each file, so a new download can be checked against the files we used.
- **Expected sample counts per split:**

| Split | Images | Image-caption pairs | Source file |
| --- | ---: | ---: | --- |
| train | 6,000 | 30,000 | `Flickr_8k.trainImages.txt` |
| validation | 1,000 | 5,000 | `Flickr_8k.devImages.txt` |
| test | 1,000 | 5,000 | `Flickr_8k.testImages.txt` |
| total | 8,000 | 40,000 | usable samples of section 3 |

- **Split unit (for example, subject, scene, document, or source video):** One image together with all five of its captions. The official files list images, and every caption follows its image into that image's split.
- **Why this unit prevents leakage:** No test image, and therefore no test image-caption pair, is seen during training or model selection, because an image and its captions are never in different splits. Two weaker forms of overlap remain.
  - **Identical caption text:** 62 of the 5,000 test captions (for 58 images) and 43 of the 5,000 dev captions (for 40 images) are identical, after tokenization, to a training caption. An example is "two dogs playing in the snow". These captions describe different images, so no pair leaks, but the model has seen their exact wording. We keep them so that the test set matches the published one, and we will note them as a limitation.
  - **Related photographs:** different photographs of the same event cannot be grouped, as explained in the duplicate check below.
- **Stratification or grouping policy:** None. We keep the split fixed by the dataset authors and do not re-split. Retrieval has no class labels to stratify on, and the release has no photographer or event identifiers to group on. The three splits still have similar caption statistics. Mean caption length is 10.79, 10.90 and 10.84 tokens for train, dev and test. Words absent from the training captions make up 1.34 percent of dev tokens and 1.22 percent of test tokens.
- **Random seed and reproducible split procedure:**
- **Duplicate / near-duplicate checks across splits:** A difference hash (dHash) of each image finds candidate pairs within 12 of 64 bits, and a candidate is confirmed when the mean absolute RGB difference at 64x64 pixels is below 5.0 on the 0 to 255 scale. These are the thresholds of our Assignment 2 audit. Of 6,701 candidates, one pair is confirmed, and both images are in the training split, so no duplicate crosses splits. The check detects copies of the same photograph, not different photographs of the same event. The release provides no photographer or event identifiers, so such related images cannot be grouped and may appear in different splits.
- **Test-set isolation plan:** The test split is used only for the final evaluation.
  - **Selection on dev:** models, hyperparameters and checkpoints are chosen on the dev split, and the test split is evaluated once, after all configurations are fixed.
  - **No test data in training:** test images and captions are never used for training, for building a vocabulary, or for computing any statistic used in preprocessing.
  - **Human judgements:** they cover only test images, so they are used only to analyse results, never for training or model selection.
  - **Retrieval pool:** at test time, each query is ranked only against the 1,000 test images or the 5,000 test captions.
- **Subset selection rules, if any:** We take no subsample. The usable set contains every image listed in the three official split files, with all five of its captions. We exclude only the 91 images that are in no split file and the five captions whose key `2258277193_586949ec62.jpg.1` names no image. We keep the very short captions, the 21 own pairs without a CrowdFlower majority yes, and the duplicate pair in the training split (section 3).

## 5. Evaluation plan

- **Primary metric(s) and why they fit the task:**
- **Secondary / per-class or per-group metric(s):**
- **Qualitative examples to inspect:**
- **Error-analysis plan:**
- **Decision criterion for comparing models:**

## 6. Model and experiment plan

- **Simple baseline:**
- **Planned pretrained or modern model(s) and source/checkpoint version:**
- **Planned fine-tuning procedure:**
- **Controlled experiment or ablation:**
  - **Hypothesis:**
  - **Changed factor:**
  - **Fixed factors:**
  - **Metric(s) and decision criterion:**

## 7. Compute and reproducibility estimate

- **Available hardware (device and memory):**
- **Estimated preprocessing, training, and evaluation time:**
- **Estimated storage and checkpoint size:**
- **Expected software framework and versions:**
- **Planned seed(s) and checkpoint selection rule:**
- **Feasibility risks and fallback within the same approved dataset/task:**

## 8. Approval record

- **Submission date and location:**
- **Instructor decision:** <!-- Approved / Approved with conditions / Rejected -->
- **Decision date:**
- **Conditions or requested changes:**
- **Changes made in response:**

---
## Appendix A. EDA evidence

All numbers and figures come from one run of `scripts/eda/run_flickr8k.py` and are stored in `results/a3/eda/flickr8k/`.

| Check | Result | Method |
| --- | ---: | --- |
| Images that fail to decode | 0 | Full decode of every image |
| Images without captions | 0 | Caption keys matched to filenames |
| Captions without an image | 0 | Same matching |
| Captions with a malformed key | 5 | Key does not end in `.jpg` |
| Images in no split file | 91 | Split files matched to filenames |
| Images listed in two split files | 0 | Pairwise overlap of split files |
| Confirmed duplicate image pairs | 1 (within train) | dHash candidates, then pixel check |
| Test captions shared by several images | 7 (13 images) | Identical text after tokenization |
| Own test pairs without a CrowdFlower majority yes | 21 of 721 | `CrowdFlowerAnnotations.txt` |

Four test images, the first four in `Flickr_8k.testImages.txt`, with their five reference captions. The examples follow this fixed rule rather than being hand-picked.

![Flickr8k test images with their five captions](../results/a3/eda/flickr8k/pair_examples.png)

Caption length has nearly the same distribution in the three splits.

![Caption length distribution by split](../results/a3/eda/flickr8k/caption_length.png)

The most frequent content words in training captions, with stop words removed, are dominated by dogs, people and colours.

![Twenty most frequent content words in training captions](../results/a3/eda/flickr8k/top_words.png)

The long side of the images is at most 500 pixels, and the short side varies widely.

![Image size distribution](../results/a3/eda/flickr8k/resolution_distribution.png)

The single confirmed duplicate pair, both images in the training split.

![Confirmed duplicate image pair](../results/a3/eda/flickr8k/duplicate_examples.png)
