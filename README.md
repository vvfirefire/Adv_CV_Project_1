
- training.txt: the training txt file listing all the training
    images and their corresponding attribute labels
- validation.txt: the validation txt file listing all the validation
    images and their corresponding attribute labels
- testset.txt: this file is generated for loading images from the private test set,
    each row of the txt file is a path to an image in the private test set
- Adv_CV_Project_1.ipynb: this is the python notebook for running the project, you can
    follow the cells in this notebook to run the project.
- All the other files are the source code for the project, and in particular main.py is
    for training and evaluating the model and predict.py is for predicting the private
    test set and generate the prediction.txt file

- prediction.txt: the prediction file for the private test set in the format of:
    image_name attribute_1_label ... attribute_40_label

One thing that need to mention is that since the original img_align_celeba zip file is too large, the submitted
zip file doesn't include this zip. You need to download the zip file and extract it and specifying the path of the
extracted folder before running the code, instructions are included in the python notebook.

The project heavily borrowed the code from https://github.com/d-li14/face-attribute-prediction

Third parties including torch, torchvision and tensorboardX are used in this project

torch: https://pytorch.org/docs/stable/torch.html
torchvision: https://pytorch.org/docs/stable/torchvision/index.html
tensorboardX: https://tensorboardx.readthedocs.io/en/latest/tensorboard.html
