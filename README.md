The fast.ai library provides many useful helper functions, resulting in a significant reduction in the number of lines of code that need to be written (e.g. in contrast to Keras). I try to summarise some of these below:

# Fast.ai Summary

## Image classification

### Simple Summary:

0. Enable data augmentation, and `precompute=True`
0. Use `lr_find()` to find highest learning rate where loss is still clearly improving
0. Train last layer from precomputed activations for 1-2 epochs
0. Train last layer with data augmentation (i.e. precompute=False) for 2-3 epochs with cycle_len=1
0. Unfreeze all layers
0. Set earlier layers to 3x-10x lower learning rate than next higher layer
0. Use `lr_find()` again
0. Train full network with `cycle_mult=2` until over-fitting

### More details:

Example usages can be found within [this notebook](https://github.com/asmith26/fastai/blob/master/courses/dl1/lesson1.ipynb):
- Choose architecture: `arch=resnet34; sz=224`
- Setup data: `PATH = "data/dogscats/"; data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))`
  - Pandas pivot_table useful for summarising how many for each class there are `label_df = pd.read_csv(label_csv); label_df.pivot_table(index='breed', aggfunc=len).sort_values('id', ascending=False)`
- Load pretrained covnets: `learn = ConvLearner.pretrained(arch, data, precompute=True)`
  - Can precompute features with `learn.precompute=True`
  - Make all layers trainable: `learn.unfreeze()`. Similarly: `learn.freeze_to(-4)`
- Train: `learn.fit(0.01, 3)`

Enhance:
- Learning rate
  - Example: `lr=np.array([1e-4,1e-3,1e-2]); learn.fit(lr, 3, cycle_len=1, cycle_mult=2)`.
    - `[1e-4,1e-3,1e-2]` indicates to use different learning rates for different layers: first few layers=1e-4, middle layers=1e-3, and FC layers=1e-2. (We refer to this as differential learning rates, although there's no standard name for this techique in the literature that we're aware of.)
    - `cycle_len` indicates how many times to reset the learning rate (search notebook for stochastic gradient descent with restarts (SGDR)).
    - `cycle_mult` indicates how we reduce the learning rates of epochs.
  - Use `lr_find()` to find highest learning rate where loss is still clearly improving.
    - Plot learning rate schedule (NOTE this changes over the course of one epoch and this plot shows that) `learn.sched.plot_lr()`
    - Plot of loss versus learning rate to see where our loss stops decreasing: `learn.sched.plot()`
- Data augmentation: pass aug_tfms (augmentation transforms) to tfms_from_model, with a list of functions to apply that randomly change the image however we wish. For photos that are largely taken from the side (e.g. most photos of dogs and cats, as opposed to photos taken from the top down, such as satellite imagery) we can use the pre-defined list of functions `transforms_side_on`.
  - See the defined function `get_augs` to visualise.
  - Test (or inference) time: use `log_preds,y = learn.TTA(); probs = np.mean(np.exp(log_preds),0)

Evaluate:
- Analysing results (few (in)correct, uncertain, most (in)correct): see the defined functions `rand_by_mask`, `rand_by_correct`, `plot_val_with_title`, `most_by_mask`, `most_by_correct`,...
  - See plot_confusion_matrix`
- TODO: add notes on from [CAM notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-CAM.ipynb)

Save:
- Save and load with `learn.save('model_name)` and `learn.load('model_name')`.

TODO: add notes for `ConvLearner.from_model_data(...)` [notebook7-cifar10](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson7-cifar10.ipynb) ("part2 is likely to discuss how to adapt the fast.ai library to your own models."))

# TODO: Structured and time series data
[Notebook](https://github.com/fastai/fastai/blob/master/courses/dl1/lesson3-rossman.ipynb)

# Fast.ai Library files
The following table was originally copied from the [fast.ai forumn (3rd Jan 2018)](http://forums.fast.ai/t/fastai-library-notes/7463):

<table/>
<tr/>
    <th/>File name</th/>
    <th/>High level summary</th/>
</tr/>
<tr/>
    <td/>imports.py</td/>
    <td/>Class which loads all external libraries e.g matplot lib etc</td/>
</tr/>
<tr/>
    <td/>transforms.py</td/>
    <td/>Classes for doing image related transformations e.g resize, reshape, crop etc</td/>
</tr/>
<tr/>
    <td/>learner.py </td/>
    <td/>Class representing a general learner. All other types of learner extend this class. e.g. CNN extends this and is coded in conv_learner.py</td/>
</tr/>
<tr/>
    <td/>conv_learner.py</td/>
    <td/>Class representing Convulation network learner</td/>
</tr/>
<tr/>
    <td/>core.py</td/>
    <td/>Defines general methods like convert to Tensor, One hot encode , Convert to GPU etc.</td/>
</tr/>
<tr/>
    <td/>model.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>dataset.py </td/>
    <td/>Class representing different types of datasets. e.g FileDataset , ImageDataset etc.</td/>
</tr/>
<tr/>
    <td/>sgdr.py</td/>
    <td/>Class for stochastic gradient descent with restarts (SGDR), a variant of learning rate annealing, which gradually decreases the learning rate as training progresses</td/>
</tr/>
<tr/>
    <td/>plots.py</td/>
    <td/>Utilities to do different types of plots. e.g Confusion matrix , Most by incorrect etc</td/>
</tr/>
    <tr/>
    <td/>layer_optimizer.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>initializers.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>torch_imports.py</td/>
    <td/>Torch library related imports and imports of pre-trained models like resnet etc.
</td/>
</tr/>
<tr/>
    <td/>metrics.py</td/>
    <td/>Metrics like accuracy, threshold etc</td/>
</tr/>
<tr/>
    <td/>io.py</td/>
    <td/>Input output utility to download files</td/>
</tr/>
<tr/>
    <td/>losses.py</td/>
    <td/>Loss functions in torch</td/>
</tr/>
<tr/>
    <td/>nlp.py</td/>
    <td/>Utilities for working with text data / NLP</td/>
</tr/>
<tr/>
    <td/>lm_rnn.py</td/>
    <td/>Different types of RNN learners e.g Sequential, Multi batch , Linear</td/>
</tr/>
<tr/>
    <td/>structured.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>set_spawn.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>rnn_train.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>rnn_reg.py</td/>
    <td/></td/>
</tr/>
<tr/>
    <td/>utils.py</td/>
    <td/></td/>
</tr/>
</table/>

