# ImageNet Testbed
This repository a large testbed to examine the robustness of various ImageNet classifers on many synthetic and natural distribution shifts. 
It is associated with the paper [Measuring Robustness to Natural Distribution Shifts in Image Classification](https://modestyachts.github.io/imagenet-testbed/).

This testbed currently supports 204 ImageNet models and 213 different evaluation settings. All the evaluation code and data used to generate the results from the paper and website can be found here. **More importantly, this repository is designed to be extremely simple to add additional models and datasets, so that future researchers may leverage this existing evalation framework to compare and evaluate their progress.**


## Installation
Dependencies: python 3.7, `requirements.txt`, and (if using gpu) cuda 10.2 capable gpus

For example, if using anaconda, create an environment and install the requirements:
```
conda create --name robustness python=3.7
conda activate robustness
pip install -r requirements.txt
```


## Running evaluations
To run models `resnet50` and `densenet121` on evaluation settings `val` (ImageNet validation set) and `imagenetv2-matched-frequency` with gpu ids `0` and `1`, run:
```
python eval.py --gpus 0 1 --models resnet50 densenet121 --eval-settings val imagenetv2-matched-frequency
```
To see a full list of models or eval settings available to run, run `python db.py --list-models-registry` or `python db.py --list-eval-settings-registry`. 
For more information on each, please see Appendices E and F in [our paper](https://arxiv.org/abs/2007.00644).


## Viewing results
We recommend our [interactive website](http://imagenet-testbed-2088145982.us-west-2.elb.amazonaws.com/) as a starting point to explore the data in this testbed.

Full results are [in the results csv](robustness_top1s.csv) or can be individually queried from the database via the `db.py` script:
```
python db.py --print-eval resnet50 objectnet-1.0-beta
```
To view all the results in the testbed at a glance, the grid can be generated via:
```
cd plotting && python paper_appendix_grid.py
```
The `plotting` directory contains the code used to generate all the plots in the paper. In particular, running `bash paper_plots.sh` will generate all the main plots in the main text of the paper (details in the file comments).

To see a full list of models or eval settings available to query for results, run `python db.py --list-models-db` or `python db.py --list-eval-settings-db`. This list is slightly longer than the registry list as some models as unavailable as part of the testbed at this time.


## Adding Custom Models
Adding a new model to the testbed only requires a few lines of code. Define a new file `src/models/new_model.py` and add your model to the `registry`. For example, here is the definition of a `resnet50`:
```
from torchvision.models import resnet50
from registry import registry
from models.model_base import Model, StandardTransform, StandardNormalization

registry.add_model(
    Model(
        name = 'resnet50',
        transform = StandardTransform(img_resize_size=256, img_crop_size=224),
        normalization = StandardNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        classifier_loader = lambda: resnet50(pretrained=True),
        eval_batch_size = 256,

        # OPTIONAL
        arch = 'resnet50',
        classify = lambda images, model, class_sublist, adversarial_attack: # your logic here
    )
)
```
The arguments shown here are required args. Note that the `transform` must only be used to bring the image to the [0, 1] domain, as all the testbed corruptions operate in that space. If your model requires further normalization, specify that as shown. Also specify `classifier_loader`, a function that will generate the model when called, and define a suitable batch size for your model.

If your model requires custom logic separate from the normal pytorch forward pass, you can specify a `classify` function to your model definition. If you specify this function, the testbed will run your custom function instead. Make sure to specify the parameters `images` and `model` (`images` will be a batch of images having gone through the transform, corruptions, and normalization, and `model` will be the return value from `classifier_loader`). There are additional optional parameters `class_sublist`, for when a model's logits are only evaluated over specific classes, and `adversarial_attack`, which defines a model's behavior under an adversarial attack. For an example of how these parameters may be used, look at `src/models/smoothing.py`.

Once you've added your model and have run the `eval.py` script to verify your results, you can persist the results in the db. First, add your model to the db with `python db.py --add-model model`. Then, run `eval.py` with the `--db` flag: `python eval.py --gpus 0 --eval-settings val --models model --db`.


## Adding Custom Evaluation Settings
Adding a new evaluation setting to the testbed also takes just a few lines of code. Define a new file `src/eval_settings/new_eval_setting.py` and add the eval setting to the `registry`. For example, here is how to add an evaluation setting for the standard `val` set:
```
from registry import registry
from eval_settings.eval_setting_base import EvalSetting, StandardDataset, accuracy_topk

registry.add_eval_setting(
    EvalSetting(
        name = 'val',
        dataset = StandardDataset(name='val'),
        size = 50000,

        # OPTIONAL
        perturbation_fn_cpu = lambda image: # your logic here
        perturbation_fn_gpu = lambda images, model: # your logic here
        metrics_fn = lambda logits, targets, image_paths: # your logic here
        class_sublist = # list of idxs of specific classes to evaluate on
        adversarial_attack = # dict of parameters used when attacking a model
    )
)
```
Evaluation settings can also be heavily customized. If you don't want to use an existing dataset from the testbed, specify `StandardDataset(path='path_to_dataset_root')` instead (formatted with inner folders grouping images for each class - if not the case, you may need to write your own `metrics_fn` utilizing `image_paths`). If you would like to calculate metrics different from the standard top1 and top5, specify a `metrics_fn` that must accept `logits` and `targets` but also optionally the `image_paths`. 

You can also specify corruptions on top of your dataset. Specify `perturbation_fn_cpu` to perturb a single image at a time on data loading threads (no gpu access), or specify `perturbation_fn_gpu` to perturb a batch at once on the gpu. `perturbation_fn_gpu` also includes the model as parameter, which is useful for perturbations such as adversarial attacks.

Once you've added your eval setting and have run the `eval.py` script to verify your results, you can persist the results in the db. First, add your eval setting to the db with `python db.py --add-eval-setting eval-setting`. Then, run `eval.py` with the `--db` flag: `python eval.py --gpus 0 --models resnet50 --eval-settings eval-setting --db`.


## Downloading Testbed Data
Our testbed also stores all of the prediction data for each model on each evaluation setting (a total of 10^9 model predictions). This data can be retrieved with the `db.py` script by specifying a particular evaluation setting and a directory to dump the results to:
```
python db.py --save-logits eval-setting /tmp/logits_dir
```


## Citation
```
@misc{taori2020measuring,
    title={Measuring Robustness to Natural Distribution Shifts in Image Classification},
    author={Rohan Taori and Achal Dave and Vaishaal Shankar and Nicholas Carlini and Benjamin Recht and Ludwig Schmidt},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2020},
    url={https://arxiv.org/abs/2007.00644},
}
