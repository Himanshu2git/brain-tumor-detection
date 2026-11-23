# Brain Tumor Classification Project — Study Guide and Viva Preparation

Himanshu Singh · ABES Engineering College

---

# PART ONE: FIRST PRINCIPLES

---

## 1. Convolution: why not a plain neural network

A 299×299 RGB image is 268,203 numbers. A fully connected first layer over that
input would need hundreds of millions of parameters, and it would still be wrong
in principle: it treats pixel (5,5) as unrelated to pixel (5,6), when
neighbouring pixels are precisely what carries the structure.

A convolution solves both problems with one construction: a small window of
weights that slides across the image.

Take a 3×3 grid of weights, called a filter or kernel. Place it over the top-left
3×3 patch. Multiply each weight by the pixel beneath it, sum the nine products,
write the result into an output grid. Slide one pixel right and repeat until the
whole image is covered.

Two properties follow.

**Locality.** Each output value depends only on a small neighbourhood, which
matches the structure of images.

**Weight sharing.** The same nine weights are applied at every position. The
layer has nine parameters rather than millions, and a pattern learned at one
location is detected everywhere.

### What a filter learns

Suppose the nine weights are negative in the left column, zero in the middle, and
positive in the right. Where the image is dark on the left and bright on the
right, the products sum to a large positive number. Where the image is uniform,
they sum to approximately zero.

That filter is a vertical edge detector. Nobody designed it. Gradient descent
arrives at it because edges are useful for the task.

### The hierarchy

Stacking convolutional layers produces a progression:

- Layer 1: edges, at various orientations
- Layer 2–3: corners, junctions, simple textures
- Layer 8–15: shapes, repeated motifs, object parts
- Layer 30+: complex configurations, e.g. a bright circumscribed mass adjacent to
  the skull

This progression from simple to complex is what "hierarchical feature learning"
means, and it is what replaced handcrafted feature engineering.

---

## 2. Depthwise separable convolution and Xception

The picture above was simplified to one filter on one channel. Real layers
operate in depth.

A colour image has 3 channels. After a layer produces 64 filter responses, the
next layer receives 64 channels. A standard convolution there is not a 3×3
window on one grid; it is a 3×3×64 volume that slides across all 64 channels
simultaneously, summing everything into one output value. Producing 128 output
channels requires 128 such filters.

Parameter count: 3 × 3 × 64 × 128 = **73,728**.

Observe what that operation does. It mixes two distinct things at once:

- **Spatial** information — the 3×3 neighbourhood
- **Cross-channel** information — combining the 64 input channels

Chollet's argument in the Xception paper is that these two jobs are sufficiently
independent that performing them separately loses very little and costs far less.

**Depthwise convolution.** Apply one 3×3 filter to each channel independently. No
mixing across channels. Cost: 3 × 3 × 64 = **576** parameters. Spatial filtering
only.

**Pointwise convolution.** Apply a 1×1 convolution spanning all 64 channels. The
window is a single pixel, so it performs no spatial operation at all; it only
combines channels. Producing 128 outputs costs 1 × 1 × 64 × 128 = **8,192**
parameters.

Total: **8,768** versus **73,728**. Roughly one-eighth the parameters for
approximately the same representational work.

"Xception" abbreviates "Extreme Inception." Inception modules split processing
into parallel branches; Xception carries that splitting to its limit by fully
separating spatial and channel operations.

### Why this mattered in your project

Your model has 20,815,148 parameters and you have 6 GB of VRAM. You measured
4.24 GB peak at batch size 16. A comparable-depth network using standard
convolution would have been several times larger, forcing batch size 4 and a
substantially longer fine-tuning run.

The layer `act4` is the ReLU following the final separable convolution block. Its
output is 2048 channels at 10×10 spatial resolution — the shape you observed when
verifying the Grad-CAM hooks. The 10×10 arises because Xception downsamples
299×299 by roughly 32× through strided convolutions.

---

## 3. Transfer learning and the normalization correction

You have 4,480 training images and 20.8 million parameters. That ratio should
fail. A model with that capacity can memorize 4,480 images and generalize
nothing.

It works because you did not start from random weights.

### What ImageNet pretraining provides

The loaded weights were trained on 1.2 million photographs across 1000 everyday
categories. Nothing medical. But recall the hierarchy: early layers learned edge,
texture, and corner detectors. Those are not properties of dogs or cars. They are
properties of *images*. A vertical edge in a photograph and a vertical edge at a
tumour margin are the same local pattern.

Transfer learning reuses this general visual machinery and re-specializes only
what is task-specific.

Your two runs measured exactly this:

| Configuration | Trainable params | Validation accuracy |
|---|---|---|
| Frozen backbone | 8,196 | 0.8804 |
| Fine-tuned | 20,815,148 | 0.9920 |

The frozen result is a direct measurement of how far generic photographic
features get you on MRI: 88%, from a single linear layer on features that never
saw a brain scan. The 11.2-point gap is the value of adaptation.

### The normalization issue

Networks are sensitive to input scale. Weights are tuned for inputs in a
particular range; supply a different range and every activation shifts, so the
learned features no longer mean what they meant.

Standard practice normalizes each channel by subtracting a mean and dividing by a
standard deviation. **The constants must match those the weights were trained
with.**

Two conventions exist:

- **PyTorch / ImageNet**: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225)
- **TensorFlow**: pixels mapped to [−1, 1], equivalent to mean 0.5 and std 0.5

Your Xception weights were ported from a TensorFlow implementation and expect
0.5. The ImageNet constants were written into `dataset.py` by default. The error
was caught by printing `default_cfg` before training, which reported
`mean/std: (0.5, 0.5, 0.5)`.

**Why this class of bug is dangerous.** Nothing errors. Shapes are correct, loss
decreases, training completes, and accuracy is a few points lower than it should
be. There is no signal that anything is wrong.

**Generalizable lesson.** Read the metadata shipped with pretrained weights
rather than assuming defaults. Every `timm` model carries a `default_cfg`
specifying input size, normalization constants, and classifier layer name.

---

## 4. The training loop

The four essential lines:

```python
optimizer.zero_grad()
loss = criterion(logits, y)
loss.backward()
optimizer.step()
```

### Logits and softmax

The network outputs four numbers per image. These are **logits**: unbounded real
numbers, not probabilities.

Softmax converts them. Exponentiate each, then divide by their sum.
Exponentiating makes everything positive; dividing by the sum makes them total 1.

Logits [2.0, 0.5, −1.0, 0.1] become approximately [0.72, 0.16, 0.04, 0.11].

### Cross-entropy loss

You need one number measuring how wrong the prediction is. If the true class
received probability *p*, the loss is **−log(p)**.

| p | −log(p) |
|---|---|
| 1.00 | 0.00 |
| 0.90 | 0.11 |
| 0.50 | 0.69 |
| 0.10 | 2.30 |
| 0.01 | 4.61 |

The loss is mild when roughly right and grows without bound as the assigned
probability approaches zero. It punishes *confident wrongness* severely.

This connects to your calibration section: cross-entropy is itself a calibration
objective. Your model nonetheless ended up overconfident, which is the phenomenon
Guo et al. documented.

### Backpropagation

You have one number (the loss) and 20.8 million parameters. For each parameter
you need to know whether increasing it raises or lowers the loss, and by how
much. That is the gradient.

Computing 20.8 million derivatives independently would be intractable.
Backpropagation computes all of them in a single backward sweep using the chain
rule. Working backward from the output, each layer receives the gradient of the
loss with respect to its output, then computes two things: the gradient with
respect to its own weights, and the gradient to pass down to the layer below.

The cost is roughly one forward pass. That efficiency is what makes deep learning
feasible.

### The four lines

**`optimizer.zero_grad()`** — PyTorch *accumulates* gradients into `.grad`
instead of overwriting. This is deliberate; it permits splitting a large batch
across several passes. But without clearing, batch 2's gradients add to batch
1's, and by batch 100 the update direction reflects every batch ever seen. The
model still trains, badly, with no error raised.

**`loss.backward()`** — runs the backward sweep, populating `.grad` for every
parameter with `requires_grad=True`.

**`optimizer.step()`** — updates the weights. Plain gradient descent would be
`w ← w − lr × w.grad`. Adam is adaptive: it maintains a running average of each
parameter's recent gradients (momentum) and of their squared magnitudes, then
scales each parameter's step by its own gradient history. Parameters with
consistently large gradients take smaller steps; those with small or noisy
gradients take relatively larger ones. This is why Adam works without extensive
learning-rate tuning.

### Why two learning rates

**Frozen, 1×10⁻³.** Only a randomly initialized linear layer trains. It must move
quickly and there is nothing to damage.

**Fine-tuned, 1×10⁻⁴.** A randomly initialized head produces large gradients
initially, and those propagate back into the pretrained backbone. At 1×10⁻³ the
first few hundred steps would substantially overwrite features that required
ImageNet-scale training to acquire. This failure mode is called **catastrophic
forgetting**.

### `model.train()` versus `model.eval()`

Two layer types behave differently between modes.

**Dropout** randomly zeroes a fraction of activations during training as
regularization. It must be disabled at inference, or predictions become random.

**Batch normalization** normalizes activations using the current batch's
statistics during training, but uses accumulated running averages at inference.
This matters because at test time the batch may contain a single image, and
normalizing by one sample's own statistics is meaningless.

Xception uses batch normalization heavily. Omitting `model.eval()` produces
plausible but incorrect validation numbers. This is why `run_epoch` sets the mode
according to whether an optimizer was supplied.

---

## 5. Three splits and early stopping

### Why training accuracy is useless

A model with 20.8M parameters can memorize 4,480 images. Training accuracy
measures memorization plus learning, and the two cannot be separated. So data is
held out; performance there measures what generalizes.

### Why two splits are insufficient

You must decide when to stop training. Too few epochs and the model is
undertrained; too many and it memorizes. You need a signal, so you train until
held-out loss stops improving and keep the best checkpoint.

**But the moment a held-out set is used for a decision, it stops being held out.**

You selected epoch 4 *because* it scored best on validation. Part of that score
is genuine quality; part is luck — epoch 4 happened to suit those particular
1,120 images. By taking the maximum over 9 attempts, you selected for both.

This is what your gap measures:

| Set | Accuracy | Used for selection? |
|---|---|---|
| Validation | 0.9920 | Yes |
| Test | 0.9487 | No |

Both were unseen during training. The only difference is selection.

**This mechanism explains a substantial share of inflated accuracies in the
literature.** Not fraud. Someone trains, watches validation, picks the best
epoch, and reports it as "accuracy." Or worse, monitors the test set while
tuning. The number is real; it does not mean what it appears to.

### Design details

**Stratified splitting.** A random 80/20 split could allocate 1,150 gliomas to
train and 250 to validation by chance. Stratified sampling splits each class
independently, preserving balance in both partitions.

**Fixed seed.** `random_state=42` makes the split identical across runs. Without
it, comparing your frozen and fine-tuned runs would be confounded by different
training data. With it, the only difference between runs is what you changed.

**Patience 5, not 1.** Validation loss is noisy. In your frozen run, epochs 7, 8,
and 9 all failed to improve; epochs 10 and 11 then both improved. Patience 1
would have stopped at epoch 7 and discarded a better model.

**Loss, not accuracy.** Accuracy is a step function — it changes only when a
prediction crosses the argmax boundary. A model improving from 51% confident to
89% confident on the correct class shows no accuracy change at all. Loss sees
that improvement, making it an earlier and smoother signal.

---

## 6. Grad-CAM

**The question:** the model said "glioma." Which regions drove that?

### The setup

At `act4` the representation is a 2048×10×10 tensor: 2048 channels, each a 10×10
spatial map. Each channel is a learned feature detector, and the 10×10 grid
preserves *where* each feature fired.

The information you want is present. The difficulty is identifying which of the
2048 channels mattered for this particular prediction.

### The gradient trick

You already have a tool that answers "how much did X influence Y": the gradient.

Take the logit for the predicted class. Compute its gradient with respect to the
`act4` feature maps. This gives, for every channel at every spatial position, how
much increasing that activation would increase the class score.

Note this differs from training. In training you differentiate the *loss* and
update weights. Here you differentiate a *class logit* and update nothing — you
only read the gradients.

### Three steps to a heatmap

**1. Average each channel's gradients over the 10×10 grid.** This collapses
2048×10×10 into 2048 numbers: one importance weight per channel. The spatial
dimension is averaged away because the question is "is this *feature* important,"
not "is it important at this exact position." Spatial information is retained in
the activations.

**2. Weighted sum of the activation maps.** Multiply each channel's 10×10 map by
its importance weight and sum. Channels that mattered contribute strongly. Result:
one 10×10 map.

**3. ReLU, then upsample.** ReLU discards negative values, retaining only regions
that *support* the class rather than argue against it. This is what makes the map
class-discriminative. Bilinear interpolation then resizes to 299×299.

### The hooks

PyTorch does not retain intermediate activations or gradients; they are computed
and discarded. A **forward hook** fires when data passes through `act4` and
captures its output. A **backward hook** fires during backpropagation and
captures the gradient. Both are required, and both returned matching
`[batch, 2048, 10, 10]` shapes when verified.

### The in-place ReLU error

`act4` overwrites its input to save memory. The backward hook wraps the layer's
output to intercept gradients; an in-place operation modifying that wrapped
tensor would corrupt the gradient path. PyTorch refused rather than returning
silently incorrect gradients. Setting `inplace=False` costs one tensor allocation
and changes nothing mathematically.

### Why heatmaps are blobby

They are computed at 10×10 and upsampled roughly 30×. The finest structure
Grad-CAM can represent is approximately 30×30 pixels.

This is a hard resolution limit, and it explains your results: IoU against tumour
masks was 0.1357 while lift was 4.66×. IoU penalizes the shape mismatch; lift
measures whether the correct region was found. Both are true; only lift answers
the question being asked.

---

## 7. Calibration

A softmax output is useful as a decision aid only if it corresponds to something
real.

**The question calibration asks:** collect every prediction made at approximately
0.92 confidence. Are 92% of them correct?

If yes, the model is calibrated and the number is usable. If only 70% are
correct, the confidence is misleading.

### Expected Calibration Error

1. Sort all predictions into bins by confidence (you used 15 equal-width bins)
2. In each bin, compute mean confidence and observed accuracy
3. Take the absolute difference
4. Average these differences, weighted by bin population

Perfect calibration gives 0. Your results:

| Set | Accuracy | Mean confidence | ECE | Gap |
|---|---|---|---|---|
| In-distribution | 0.9487 | 0.9795 | 0.0325 | +0.0308 |
| External | 0.8983 | 0.9163 | 0.0227 | +0.0180 |

A **positive gap** means overconfidence: the model claims more certainty than it
earns. Both are positive but small.

### Why the external set is better calibrated

This inverts the standard expectation, and the bin occupancies explain it.

In distribution, **1,490 of 1,600 predictions** fall in the top bin at mean
confidence 0.996. The model is effectively binary — near-certain about almost
everything. Confidence therefore carries almost no information, and the 5% of
errors occur at high confidence.

Externally, only 4,303 of 6,056 fall in the top bin. The remainder spread across
the middle of the range, with accuracy tracking confidence closely:

| Confidence bin | Accuracy |
|---|---|
| 0.99 | 0.979 |
| 0.90 | 0.834 |
| 0.77 | 0.700 |
| 0.70 | 0.652 |

That is close to monotonic and genuinely actionable — a referral threshold could
be set on it. Confidence becomes *more* informative outside the training
distribution, not less.

### A caution about MCE

Maximum calibration error was 0.4365 in distribution. It arises from a bin
containing **three predictions**, all incorrect. With three samples the accuracy
estimate is meaningless. MCE should be reported with bin occupancy or not at all.

---

## 8. Evaluation methods you built

### Provenance verification by hashing

Most public four-class brain MRI datasets are recombinations of the same three
sources. A dataset with a different name and DOI may still contain the same
files. Evaluating on such a dataset is testing on training data.

**Your procedure:**

1. Compute a SHA-256 digest of every training image and every candidate image.
   Any intersection disqualifies the candidate. A cryptographic hash is either
   equal or it is not; there is no threshold to argue about.
2. For a random sample, compute normalized intensity-signature correlation
   against all training images. This catches re-encoded or resized copies, which
   hash differently but are the same underlying scan.
3. Compare the maximum correlation against the empirical noise floor for
   unrelated brain MRI pairs (approximately 0.75, established during the failed
   provenance recovery).

The Bangladesh dataset passed both: zero byte overlap across 6,056 images, and
maximum correlation 0.907 with no pair above 0.99.

### The intensity signature

Downsample both images to 32×32, then normalize to zero mean and unit variance.

**Why downsample.** JPEG compression discards high-frequency detail. Resizing
does the same physically. What survives is coarse intensity layout. Comparing at
full resolution would ask one side to match detail it no longer contains.

**Why normalize.** Zero-mean unit-variance makes the comparison invariant to
brightness and contrast shifts. After normalization, correlation is simply a dot
product, so the entire 7,200 × 3,064 comparison is one matrix multiplication.

### The prediction flip rate

In the robustness sweeps you measured two things per perturbation level:
accuracy, and the fraction of predictions that *changed* relative to the
unperturbed baseline.

The second is the stricter measure. A model can hold accuracy constant while
churning which individual cases it gets right. For a diagnostic aid, instability
under acquisition variation is itself a defect: the same patient scanned twice
should not receive different answers.

### Grad-CAM lift

Tumours occupy 2.02% of the image on average. If attention were uniform, 2.02% of
heatmap mass would land inside the tumour by chance. You measured 10.14%.

**Lift = 10.14 / 2.02 = 4.66×**

This metric accounts directly for the size mismatch that makes IoU misleading. A
lift of 1.0 would mean the heatmap is no better than noise.

---

# PART TWO: VIVA QUESTIONNAIRE

Answers grounded in your measurements. Where a claim is uncertain, the answer
says so — examiners respect calibrated confidence more than assertion.

---

## A. Architecture and training

**Q: Why Xception rather than ResNet or VGG?**

Comparative work on this exact four-class task found Xception the strongest of six
pretrained backbones. It also suits our hardware: depthwise separable convolution
gives roughly one-eighth the parameters of standard convolution for comparable
representational capacity, which matters on a 6 GB GPU. Our frozen baseline
comparison was between frozen and fine-tuned Xception, not across architectures,
so we do not claim Xception is best — we cite prior work for that.

**Q: Explain depthwise separable convolution.**

A standard convolution mixes spatial and cross-channel information in one
operation. Depthwise separable convolution splits them: a depthwise stage applies
one filter per channel independently, handling spatial filtering only; a pointwise
1×1 stage combines channels without any spatial extent. For 64→128 channels with
a 3×3 kernel, that is 8,768 parameters instead of 73,728.

**Q: What is transfer learning transferring?**

Low- and mid-level visual features. Early convolutional layers learn edge,
texture, and corner detectors, which are properties of images rather than of the
ImageNet categories. Our frozen-backbone result quantifies this: 88.04%
validation accuracy from a single linear layer on features that never saw a brain
scan. Fine-tuning adds 11.2 points by adapting those features to MRI.

**Q: Why did you use two different learning rates?**

The frozen configuration trains only a randomly initialized linear layer, so
1×10⁻³ is appropriate. In fine-tuning, gradients from that head propagate into
the pretrained backbone, and at 1×10⁻³ the early steps would overwrite features
acquired from ImageNet-scale training — catastrophic forgetting. We used 1×10⁻⁴,
which is standard for fine-tuning.

**Q: Why batch size 16 and not 32?**

Memory. A fine-tuning step at batch 16 peaks at 4.24 GB on a 6 GB card. At batch
32 the allocator reports 8.19 GB, which exceeds physical VRAM. It does not fail,
because the WSL2 driver spills to host memory over PCIe — the run completes
correctly but far more slowly. We chose the batch size that fits in real VRAM.

**Q: Why does `model.eval()` matter?**

Dropout and batch normalization behave differently between training and
inference. Dropout must be off at inference. Batch normalization uses batch
statistics during training and accumulated running averages at inference, because
a test batch may contain one image and normalizing by its own statistics is
meaningless. Xception uses batch normalization heavily, so omitting `eval()`
produces plausible but incorrect results.

---

## B. Data and protocol

**Q: Your paper's earlier draft reported 99.7%. You report 94.87%. Why the
difference?**

Three separate reasons.

First, the earlier figure came from a different dataset version. The collection
was revised in February 2026; the earlier release had 5,712 training images with
unequal class counts, ours has 5,600 balanced at 1,400 per class. The earlier
confusion matrix also implies 654 test samples where ours has 1,600. Those
results cannot be reproduced because the conditions no longer exist.

Second, our protocol uses the test set once, after all model selection is
complete. Our validation accuracy — the figure most comparable to what is usually
reported — was 99.20%.

Third, 94.87% is what remains once the number used for early stopping is set
aside. Both figures are on unseen data; only one was used to make a decision.

**Q: Why three splits instead of two?**

Because early stopping requires a held-out signal, and using a set to make a
decision biases it. We selected epoch 4 because it scored best on validation over
9 epochs; some of that score is quality and some is selection over noise. The
test set answers the question the validation set can no longer answer.

**Q: What is data leakage and is it present here?**

Leakage is any path by which test information reaches training. We excluded exact
duplicates by hashing all 7,200 files: 153 duplicate groups exist, all within a
single split, none spanning train and test.

We could not exclude patient-level leakage. The largest source component contains
about 13.2 slices per patient across 233 patients, and adjacent slices from one
patient are visually near-identical but byte-distinct. Under an image-level random
split, slices from one patient can appear in both partitions. We attempted to
recover patient identifiers by matching back to the source dataset and failed, so
we report this as an unquantified limitation.

**Q: Why did the provenance matching fail?**

Our matcher worked — it passed a positive control, returning a re-encoded source
image's own record at correlation 1.0000. On the actual data the median
best-match correlation was 0.724 with a median margin of 0.004, and class
agreement was 31.6% against a chance rate of 33%. The no-tumor class, which has
no true match in the source set, scored highest of all, establishing 0.75 as the
noise floor. The redistributed images have been altered beyond the point where
coarse intensity structure identifies them.

**Q: Why do you use 0.5 normalization when the literature says [0,1] rescaling?**

Our Xception weights were ported from a TensorFlow implementation whose
preprocessing maps pixels to [−1, 1], and the distributed model metadata
specifies mean and standard deviation of 0.5. Applying ImageNet constants would
shift the input distribution away from the one the features were learned under.
The degradation is silent — the model still trains and reports plausible accuracy
— so we verified the metadata rather than assuming the default.

---

## C. Results

**Q: Why is glioma your worst class?**

Glioma recall is 0.8275, with 69 of 400 missed and 46 assigned to meningioma.
Glioma precision, however, is 0.9940 — the model is conservative rather than
confused. The same asymmetry appears in the frozen model at 0.6975 recall, so it
is not introduced by fine-tuning.

The likely explanation is in the data. On our external dataset, glioma recall is
0.9980, the best of the three classes. The glioma class in the source collection
was substituted between releases because of labelling problems, and our results
are consistent with residual label noise. We cannot verify this without corrected
labels.

**Q: Is the direction of that error clinically acceptable?**

No, and we say so. Glioma is the more aggressive presentation, so a false negative
carries greater consequence than a false positive. Shifting the operating point
to raise glioma recall at the cost of precision would be a reasonable next step.

**Q: How did you verify the external dataset is genuinely external?**

Two tests. Byte-level: SHA-256 digest of every image on both sides, zero
intersection across 6,056 images. Correlation-level: 300 random candidate images
matched against all 5,600 training images, maximum correlation 0.907 with no pair
above 0.99, both within the noise floor for unrelated brain MRI.

We verified rather than assumed because most public datasets on this task are
recombinations of the same three sources, and evaluating on one of those would be
testing on training data.

**Q: Why report two external accuracy figures?**

They answer different questions. The unrestricted figure (89.83%) permits the
model to output "no tumor," a class with no instances in that dataset, so every
such prediction counts as an error. That is the realistic deployment number. The
restricted figure (93.39%) removes that output and isolates discrimination among
the three classes present.

**Q: 269 images were predicted "no tumor" when every image contained a tumor.
What does that mean?**

It identifies the model's characteristic out-of-distribution failure: on an
unfamiliar acquisition it reports absence rather than confusing one tumour type
with another. For a screening application this is the most consequential error
available, and it is not observable in any in-distribution evaluation.

---

## D. Explainability

**Q: How does Grad-CAM work?**

It takes the gradient of a class logit with respect to the final convolutional
layer's feature maps. Those gradients are averaged over the spatial dimensions to
give one importance weight per channel. The activation maps are then combined as
a weighted sum using those weights, passed through ReLU to retain only positive
contributions, and upsampled to the input resolution.

**Q: Why hook `act4` specifically?**

It is the last layer that retains spatial structure. Its output is 2048 channels
at 10×10; after global pooling the spatial information is gone. It is also the
post-activation output, matching the original Grad-CAM formulation.

**Q: Your IoU is only 0.1357. Doesn't that mean Grad-CAM failed?**

No, and this is why we report three metrics rather than one. Grad-CAM is computed
at 10×10 and upsampled roughly 30×, so its finest representable structure is
about 30×30 pixels, while a tumour mask is a tight boundary. IoU is bounded low
by that resolution limit regardless of whether the model is looking in the right
place.

The metric that answers the actual question is lift. Tumours occupy 2.02% of the
image; Grad-CAM places 10.14% of its attention there, a lift of 4.66× over
uniform attention. Mean and median lift are close, so this is a general property
rather than a favourable minority.

**Q: How did you obtain ground-truth masks?**

The BRISC annotation set provides expert-reviewed pixel-level masks for images
from the same source collection. Those images are byte-identical to ours, so we
linked them by exact SHA-256 digest rather than approximate matching. 4,725 of
4,793 matched; 962 fall in our test split and 949 have usable masks. BRISC's
independently assigned class labels agreed with ours on all 949, confirming the
linkage.

**Q: Why does localization vary by tumour class?**

Meningioma localizes best on all three measures (lift 5.93×, IoU 0.2545),
pituitary worst by IoU (0.0535). This ordering is consistent with the
radiological character of the presentations: meningiomas are extra-axial and
typically well-circumscribed; pituitary tumours sit at the skull base among other
enhancing structures; gliomas are intra-axial and infiltrative with indistinct
margins.

We report this as an observed correspondence, not a causal claim. We measured
properties of the model's attention, not of the lesions.

**Q: You say the annotated subset is biased. Explain.**

The 949 linked images contain 4 of our 69 glioma errors, a rate of 1.6%. The 147
test gliomas not in the annotation contain 65 of them, a rate of 44%. The subset
is therefore not a random sample with respect to classification difficulty.

The annotation was produced for segmentation, not for benchmarking, so this is
not a criticism of it. The consequence for us is that our localization results
characterize behaviour on the easier portion of the test set and should be read
as an upper bound.

---

## E. Calibration and robustness

**Q: What is expected calibration error?**

Predictions are grouped into confidence bins. Within each bin, mean confidence is
compared with observed accuracy, and the bin-weighted mean of the absolute
differences is reported. Perfect calibration gives zero. Ours is 0.0325 in
distribution and 0.0227 externally, both mildly overconfident.

**Q: Why is calibration better on external data? That seems backwards.**

Saturation. In distribution, 1,490 of 1,600 predictions sit in the top bin at
0.996 confidence, so the model has almost no scope to be well calibrated against
94.87% accuracy, and the errors it makes are made confidently. On the external
set the predictions spread across the confidence range with accuracy tracking
confidence closely — 0.979 at the 0.99 bin, 0.834 at 0.90, 0.700 at 0.77.
Confidence is more informative outside the training distribution, not less.

**Q: What does the noise result show?**

Accuracy is stable under brightness, contrast, rotation, and translation. Under
additive Gaussian noise it degrades sharply: 15 points lost at σ = 0.02, which is
1% of the input range, and chance-level accuracy at σ = 0.05.

More concerning is the confidence trajectory. Confidence falls with accuracy as
far as σ = 0.03, reaching 0.8709, then rises to 0.9806 at σ = 0.05 and 0.9967 at
σ = 0.1 — where accuracy is at chance. The highest confidence across the entire
sweep coincides with the lowest accuracy.

**Q: Why does that matter practically?**

It indicates a bounded region of trustworthiness. Within the range the model can
accommodate, confidence declines with accuracy and is usable. Past a threshold the
relationship inverts, and there is no accompanying signal that the boundary has
been crossed. MRI noise characteristics vary with field strength, coil
configuration, and acquisition time, so this is not solely an adversarial concern.

**Q: Why report flip rate as well as accuracy?**

Accuracy can remain constant while the specific cases the model gets right change
underneath it. For a diagnostic aid, instability under acquisition variation is
itself a defect — the same patient scanned twice should not receive different
answers.

---

## F. Difficult questions

**Q: What is novel here? Xception and Grad-CAM are both standard.**

The architecture is not novel and we say so in the introduction. What we
contribute is the evaluation: external validation with verified provenance,
single-use test protocol, calibration measured in and out of distribution,
Grad-CAM grounded quantitatively against expert masks, and a documented curation
effect in the annotated subset. We did not find prior work on this task that
verifies the independence of a claimed external validation set.

**Q: Is this clinically usable?**

No. We state that explicitly. It has not been prospectively evaluated, has no
specialist assessment, has an unfavourable error direction on the most aggressive
tumour class, cannot be shown free of patient-level leakage, and has a
demonstrated failure mode in which it reports absence on unfamiliar acquisitions
while remaining confident. It is a research prototype.

**Q: Your sample size for misclassified cases is very small.**

Yes. The linked subset contains 11 errors total, which is why we did not attempt
a statistical comparison between correct and incorrect predictions. We report the
composition of the subset instead, which is itself informative, and identify the
comparison as future work requiring masks for the excluded cases.

**Q: How much of this did you do yourself?**

I built and ran the entire pipeline: environment, data audit, training, both
evaluations, Grad-CAM implementation and validation, calibration, and robustness.
I used an AI assistant for code syntax and for discussing method choices, in the
same way one would use documentation and a colleague. The decisions — PyTorch
over Keras, single-use test protocol, verifying provenance before trusting the
external set, reporting lift alongside IoU — are ones I can defend, and the
results are reproducible from the commands in Appendix A.

---

## G. Quick reference: every number

| Measurement | Value |
|---|---|
| Training images | 5,600 (1,400/class) |
| Test images | 1,600 (400/class) |
| Train/val split | 4,480 / 1,120, stratified, seed 42 |
| Model parameters | 20,815,148 |
| Frozen trainable | 8,196 |
| Frozen validation | 0.8804 (epoch 15) |
| Fine-tuned validation | 0.9920 (epoch 4) |
| Frozen test accuracy | 0.8494 (macro F1 0.8456) |
| Fine-tuned test accuracy | 0.9487 (macro F1 0.9479) |
| Glioma recall / precision | 0.8275 / 0.9940 |
| External accuracy (4-class) | 0.8983 |
| External accuracy (3-class) | 0.9339 |
| External glioma recall | 0.9980 |
| False "no tumor" predictions | 269 of 6,056 |
| ECE in-distribution | 0.0325 |
| ECE external | 0.0227 |
| Grad-CAM lift | 4.66× (median 4.49×) |
| Grad-CAM IoU | 0.1357 |
| Linked mask images | 949 |
| Noise: accuracy at σ=0.02 | 0.7963 |
| Noise: accuracy at σ=0.05 | 0.2669 (confidence 0.9806) |
| Duplicate groups | 153, none crossing splits |
| Slices per patient (source) | 13.2 across 233 patients |
| Peak VRAM, batch 16 | 4.24 GB |

---

*Environment: Python 3.12.14, PyTorch 2.11.0+cu128, timm 1.0.29,
scikit-learn 1.9.0. NVIDIA RTX 4050 Laptop GPU (6 GB), Intel i5-13420H, WSL2.*
