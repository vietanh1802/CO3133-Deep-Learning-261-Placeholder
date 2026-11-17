# Assignment 3 — M1 Dataset Proposal

> Complete this proposal for one task track before beginning the main Assignment 2 implementation. Submit it for instructor approval. Do not treat this template as an approved dataset proposal.

## 1. Group and proposed task

- **Group name / ID:** Placeholder
- **Members: **
  | Full name | Student ID |
  | ---------------- | ---------- |
  | Vu Duc Viet Anh | 2352074 |
  | Tran Lam Anh | 2352067 |
  | Le Nguyen Khang | 2352470 |
  | Tran Nguyen Giap | 2352284 |

- **Task track:** Multimodal learning — image captioning.
- **Problem statement:** Generate an English sentence describing the main objects, actions and scenes in an image.
- **Prediction unit:** One image.
- **Input:** An image. The model also receives the preceding reference-caption tokens to predict the next token.
- **Output / labels:** A generated English caption. Each image has five human-written reference captions for training and evaluation.
- **Why this task and dataset are suitable:** Flickr8k links images with natural-language descriptions, allowing the model to learn the relationship between visual content and text. The proposed split contains 6,000 training, 1,000 validation, and 1,000 test images.

## 2. Dataset identity and access

- **Dataset name:** Flickr8k.

- **Source and download URL:** Original dataset information: https://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html. Download mirror: https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k.

- **Version / release / access date:** Downloaded from the mirror release tagged `Flickr8k`, published on 20 February 2019. Download date: 10 November 2026.

- **License and permitted use:** The original authors provide access for non-commercial research and educational use. Image use must respect Flickr's terms of use. The dataset authors do not own the copyright of the images.

- **Data format and storage size:** JPEG images and plain-text caption and split files. Each caption record contains an image filename, a caption ID from 0 to 4, and the caption text. The downloaded archives contain 1,115,419,746 bytes of image data and 2,340,801 bytes of text data in compressed form. Extracted JPEG files total approximately 1,063.09 MiB, and the main text files total 9.64 MiB.

- **Collection and annotation process, if known:** Images were collected from Flickr and paired with five human-written English captions per image. Captions describe the visible objects, actions, and scenes. Image filenames link each image to its captions.

- **Known limitations or biases:** The dataset focuses on people and animals performing everyday actions, which limits its coverage of other visual domains. Captions are in English and differ in wording and the details they describe. The standard split contains only 8,000 images. Our downloaded files contain 8,091 JPEG images but captions for 8,092 image IDs; all 8,000 images in the supplied train, validation, and test lists have corresponding image files and captions.

## 3. Dataset size and preliminary analysis

- **Total samples:** The downloaded data contain 8,091 JPEG images and 40,460 captions associated with 8,092 image IDs.

- **Proposed usable samples:** We propose using the 8,000 images in the supplied splits: 6,000 for training, 1,000 for validation, and 1,000 for testing. Each image has five captions, giving 40,000 image–caption pairs.

- **Annotation types and counts:** Five human-written English captions per image. The proposed subset contains 30,000 training, 5,000 validation, and 5,000 test captions.

- **Class / target distribution:** This is a sequence-generation task without fixed class labels. Targets are English captions. Word-frequency and caption-length distributions remain to be measured using the training split.

- **Input size or length distribution:** Image width, height, aspect ratio, and caption-length distributions remain to be measured. Caption length will initially be counted as whitespace-separated words, before model tokenization.

- **Missing, invalid, duplicate, or low-quality records:** All 8,000 image IDs in the supplied splits have corresponding image files and five captions. The split lists have no overlapping image IDs. The full caption file contains 8,092 image IDs, while the image archive contains 8,091 JPEG files. Image decoding, duplicate-content detection, empty-caption checks, and pair-quality inspection remain to be completed.

- **Evidence (tables, plots, or analysis links):** Preliminary counts come from the extracted image directory, `Flickr8k.token.txt`, and the three supplied split lists. Reproducible EDA outputs and figures are pending.

- **Relevant task-specific checks:** Check image decoding, image–caption filename matching, five captions per image, empty or repeated captions, caption lengths, and duplicate or near-duplicate images across splits. Inspect sample image–caption pairs for mismatches. Keep all captions of an image in the same split.

- **How the dataset meets the handbook's threshold for this track (or justification for an exception):** Section 28.2 requires approximately 5,000 or more images for captioning and prefers multiple captions per image. Our proposed subset contains 8,000 images with five captions each, including 6,000 training images, and therefore meets the default size threshold. Instructor approval is still required.

## 4. Split and leakage prevention

- **Official or custom train / validation / test split:** Use the supplied split lists: `Flickr_8k.trainImages.txt`, `Flickr_8k.devImages.txt`, and `Flickr_8k.testImages.txt`.

- **Expected sample counts per split:** Training: 6,000 images and 30,000 captions. Validation: 1,000 images and 5,000 captions. Test: 1,000 images and 5,000 captions.

- **Split unit (for example, subject, scene, document, or source video):** Image ID. All five captions associated with an image remain in the same split.

- **Why this unit prevents leakage:** Grouping by image ID prevents the same image from appearing in multiple splits through different captions. It does not guarantee separation of related scenes or near-duplicate images, which require additional checks.

- **Stratification or grouping policy:** Preserve the supplied image-level grouping without additional stratification. Captioning has no fixed class labels to stratify by.

- **Random seed and reproducible split procedure:** No new random split is generated. Split membership comes directly from the supplied lists. Record their SHA-256 checksums to identify the exact files used. Use seed 42 for training randomness.

- **Duplicate / near-duplicate checks across splits:** No overlapping image IDs were found between the supplied splits. We will check exact duplicates using file hashes and screen for near-duplicates using perceptual hashes, followed by manual inspection. Any confirmed leakage and changes to the split will be documented before training.

- **Test-set isolation plan:** Use training data only to fit the vocabulary and any learned preprocessing. Use validation data for model selection, hyperparameter choices, and decoding settings. Keep test captions out of training and tuning, and evaluate the test set only after these choices are fixed. Initial test-data inspection is limited to integrity and leakage checks.

- **Subset selection rules, if any:** Use only the 8,000 image IDs in the supplied split lists and their five captions. Exclude records outside these lists. Any further exclusion due to corruption or confirmed duplication will be documented with the affected image IDs and updated counts.

## 5. Evaluation plan

- **Primary metric(s) and why they fit the task:** CIDEr, which compares generated captions with the five reference captions using weighted n-gram similarity.

- **Secondary / per-class or per-group metric(s):** BLEU-4 and performance grouped by reference-caption length.

- **Qualitative examples to inspect:** Show images, reference captions, and predictions from each model, including successful and failed cases.

- **Error-analysis plan:** Inspect incorrect objects, actions, counts, and details unsupported by the image. Report their frequency in a fixed sample.

- **Decision criterion for comparing models:** Select checkpoints using validation CIDEr. Compare final models using test CIDEr under the same evaluation protocol, with BLEU-4, qualitative errors, and compute cost as supporting evidence.

## 6. Model and experiment plan

- **Simple baseline:** Image-only nearest-neighbor caption transfer using training images; a text-only LSTM language model; and a simple fusion model combining pooled image features with an LSTM decoder.

- **Planned pretrained or modern model(s) and source/checkpoint version:** Torchvision ResNet-50 with `ResNet50_Weights.IMAGENET1K_V2`, combined with an LSTM decoder and spatial attention.

- **Planned fine-tuning procedure:** First freeze the image encoder and train the decoder using cross-entropy and teacher forcing. Then fine-tune the encoder's final block at a lower learning rate. Select checkpoints using validation CIDEr.

- **Controlled experiment or ablation:**
  - **Hypothesis:** Spatial attention improves caption quality compared with global average pooling.
  - **Changed factor:** Replace global average pooling with learned spatial attention.
  - **Fixed factors:** Dataset splits, encoder checkpoint, decoder hidden size, training schedule, seed, and decoding settings.
  - **Metric(s) and decision criterion:** Compare test CIDEr and BLEU-4 after validation-based checkpoint selection. Higher CIDEr supports the hypothesis for this run.

## 7. Compute and reproducibility estimate

- **Available hardware (device and memory):** GPU model, VRAM, and system RAM: to be confirmed before training.

- **Estimated preprocessing, training, and evaluation time:** Pending a short benchmark. Estimate training time from measured epoch time and the planned epoch budget; measure caption generation separately.

- **Estimated storage and checkpoint size:** Approximately 2.1 GiB for the downloaded archives and extracted dataset. Additional storage for features, model weights, optimizer states, and logs will be measured after implementation.

- **Expected software framework and versions:** Python, PyTorch, and torchvision. Exact versions and caption-evaluation dependencies will be recorded and locked before experiments.

- **Planned seed(s) and checkpoint selection rule:** Seed 42 for initial experiments. Select the checkpoint with the highest validation CIDEr; keep test data out of model selection.

- **Feasibility risks and fallback within the same approved dataset/task:** Limited GPU memory and training time. Reduce batch size and use gradient accumulation if needed. Cache frozen encoder features during decoder training and reduce the training budget consistently across compared models.

## 8. Approval record

- **Submission date and location:** Not yet submitted. Planned submission: 18 November 2026, through the course-designated channel.
- **Instructor decision:** Pending.
- **Decision date:** Pending.
- **Conditions or requested changes:** Pending instructor feedback.
- **Changes made in response:** Not applicable yet.
