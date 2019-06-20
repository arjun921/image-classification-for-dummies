# Image Classifier Using Tensorflow/ Inception-v3

Tensorflow Image classifer built by transfer learning on Inceptionv3 architecture.

This method does not need a GPU and can produce very accurate results in a very short time. (depending on the number of images)


### Getting started

First off, we need to setup our project folder with all the contents required for running the train script.

Run the following commands to clone the repo and setup the basic folder structure.

```bash
git clone https://github.com/arjun921/image-classification-for-dummies.git
cd image-classification-for-dummies
mkdir inception
mkdir bottlenecks
mkdir trained_model
mkdir trained_model_serving
mkdir data
mkdir data/cat
mkdir data/dog
pipenv install --skip-lock
# if pipenv command not found, install using `pip3 install pipenv` and rerun `pipenv install`
pipenv shell
```

Split your data into train and test and put it as the following structure

> data
>
> > cat
> >
> > > Images for cats
> >
> > dog
> >
> > > Images for dogs

For ease of data availabilty, I will be using the renowned Dogs vs. Cats data set from [here](https://www.kaggle.com/c/dogs-vs-cats)

Once we have all the needed folders and the necessary data to train on, we can begin training. Beware, the data you will download has all images under one folder. You will have to copy the pictures of dogs into the `data/dog` and cat pictures to `data/cat` . Replace the foldernames and content with the images and classnames of the images you're trying to classify. The foldername 

## How To Train

Make sure Tensorflow gets imported into python without any errors.

#### Command

Execute command on terminal from within folder:

```bash
python3 tf_retrain.py --bottleneck_dir=bottlenecks/ \
--how_many_training_steps 500 \
--model_dir=inception/ \
--output_graph=trained_model/unoptimized_model.pb \
--output_labels=trained_model/labels.txt \
--image_dir=data/
```

For tensorflow serving

```bash
python3 retrain_serving.py  \
--bottleneck_dir=bottlenecks/  \
--how_many_training_steps 100  \
--model_dir=inception/  \
--output_graph=trained_model/unoptimized_model.pb  \
--output_labels=trained_model/labels.txt  \
--image_dir=train_data/train/  \
--saved_model_dir=trained_model_serving/1
```



#### Arguments

##### bottleneck_dir

- Images converted to Bottlenecks, saved in this directory. 
- Temporary
- Required only during training
- Keeping this folder reduces time in retraining with same data if classes are not going to be changed.
- Change Directory as required for new training

##### how_many_training_steps

- number of steps to run
- 4000 (Default)
- Increase if number of images increases

##### model_dir

- IMPORTANT directory containing inception model
- **Do Not Change Path, leave as is**.

##### output_graph

- Store File with .pb extension
- Change Path of graph as required
- The actual training stored in this file
- Memory of the classifier.

##### image_dir

- Point to folder with data.
- Folder should have subfolders with data in subfolders
- Directory Layout as shown Below
- Point path to ImageDirectory
- IMPORTANT! Only jpg files should be added to training folder



#### Troubleshooting

- In case of permission error, run command in admin command prompt or sudo <command>  to fix.

#### How to Improve Training

- Run Same command as Train. 
- Ensure to use old bottleneck directory to reduce time taken
- Add more images to respective class directory to improve accuracy
- Increase number of steps
- Ensure data is clean and doesn't have false positives


## Testing/Predicting the class given an image

To test the classifier against an image of your choice, run the following script with the path to as an argument. 

```bash
python3 tf_test.py <path to image> trained_model/unoptimized_model.pb trained_model/labels.txt
```

The class and its confidence will be shown on the terminal. Thats it! you're done with image classification :)

## Theory 

This isn't absolutely necessary to read, but its good to know what each argument does or what happens in the actual process of training. Read along to know more.

### Bottlenecks

The script can take thirty minutes or more to complete, depending on the speed of your machine. The first phase analyzes all the images on disk and calculates the bottleneck values for each of them. 'Bottleneck' is an informal term we often use for the layer just before the final output layer that actually does the classification. This penultimate layer has been trained to output a set of values that's good enough for the classifier to use to distinguish between all the classes it's been asked to recognize. That means it has to be a meaningful and compact summary of the images, since it has to contain enough information for the classifier to make a good choice in a very small set of values. The reason our final layer retraining can work on new classes is that it turns out the kind of information needed to distinguish between all the 1,000 classes in ImageNet is often also useful to distinguish between new kinds of objects.

Because every image is reused multiple times during training and calculating each bottleneck takes a significant amount of time, it speeds things up to cache these bottleneck values on disk so they don't have to be repeatedly recalculated. By default they're stored in the `/tmp/bottleneck` directory(unless new directory specified in argument), and if you rerun the script they'll be reused so you don't have to wait for this part again.

### Training

Once the bottlenecks are complete, the actual training of the top layer of the network begins. You'll see a series of step outputs, each one showing training accuracy, validation accuracy, and the cross entropy. The training accuracy shows what percent of the images used in the current training batch were labeled with the correct class. The validation accuracy is the precision on a randomly-selected group of images from a different set. The key difference is that the training accuracy is based on images that the network has been able to learn from so the network can overfit to the noise in the training data. A true measure of the performance of the network is to measure its performance on a data set not contained in the training data -- this is measured by the validation accuracy. If the train accuracy is high but the validation accuracy remains low, that means the network is overfitting and memorizing particular features in the training images that aren't helpful more generally. Cross entropy is a loss function which gives a glimpse into how well the learning process is progressing. The training's objective is to make the loss as small as possible, so you can tell if the learning is working by keeping an eye on whether the loss keeps trending downwards, ignoring the short-term noise.

By default this script will run 4,000 training steps. Each step chooses ten images at random from the training set, finds their bottlenecks from the cache, and feeds them into the final layer to get predictions. Those predictions are then compared against the actual labels to update the final layer's weights through the back-propagation process. As the process continues you should see the reported accuracy improve, and after all the steps are done, a final test accuracy evaluation is run on a set of images kept separate from the training and validation pictures. This test evaluation is the best estimate of how the trained model will perform on the classification task. You should see an accuracy value of between 90% and 95%, though the exact value will vary from run to run since there's randomness in the training process. This number is based on the percent of the images in the test set that are given the correct label after the model is fully trained.

### Hyper-parameters

There are several other parameters you can try adjusting to see if they help your results. The `--learning_rate` controls the magnitude of the updates to the final layer during training. Intuitively if this is smaller then the learning will take longer, but it can end up helping the overall precision. That's not always the case though, so you need to experiment carefully to see what works for your case. The `--train_batch_size` controls how many images are examined during one training step, and because the learning rate is applied per batch you'll need to reduce it if you have larger batches to get the same overall effect.

### Training, Validation, and Testing Sets

One of the things the script does under the hood when you point it at a folder of images is divide them up into three different sets. The largest is usually the training set, which are all the images fed into the network during training, with the results used to update the model's weights. You might wonder why we don't use all the images for training? A big potential problem when we're doing machine learning is that our model may just be memorizing irrelevant details of the training images to come up with the right answers. For example, you could imagine a network remembering a pattern in the background of each photo it was shown, and using that to match labels with objects. It could produce good results on all the images it's seen before during training, but then fail on new images because it's not learned general characteristics of the objects, just memorized unimportant details of the training images.

This problem is known as overfitting, and to avoid it we keep some of our data out of the training process, so that the model can't memorize them. We then use those images as a check to make sure that overfitting isn't occurring, since if we see good accuracy on them it's a good sign the network isn't overfitting. The usual split is to put 80% of the images into the main training set, keep 10% aside to run as validation frequently during training, and then have a final 10% that are used less often as a testing set to predict the real-world performance of the classifier. These ratios can be controlled using the `--testing_percentage` and `--validation_percentage` flags. In general you should be able to leave these values at their defaults, since you won't usually find any advantage to training to adjusting them.

Note that the script uses the image filenames (rather than a completely random function) to divide the images among the training, validation, and test sets. This is done to ensure that images don't get moved between training and testing sets on different runs, since that could be a problem if images that had been used for training a model were subsequently used in a validation set.

You might notice that the validation accuracy fluctuates among iterations. Much of this fluctuation arises from the fact that a random subset of the validation set is chosen for each validation accuracy measurement. The fluctuations can be greatly reduced, at the cost of some increase in training time, by choosing `--validation_batch_size=-1`, which uses the entire validation set for each accuracy computation.

Once training is complete, you may find it insightful to examine misclassified images in the test set. This can be done by adding the flag `--print_misclassified_test_images`. This may help you get a feeling for which types of images were most confusing for the model, and which categories were most difficult to distinguish. For instance, you might discover that some subtype of a particular category, or some unusual photo angle, is particularly difficult to identify, which may encourage you to add more training images of that subtype. Oftentimes, examining misclassified images can also point to errors in the input data set, such as mislabeled, low-quality, or ambiguous images. However, one should generally avoid point-fixing individual errors in the test set, since they are likely to merely reflect more general problems in the (much larger) training set.



## Visualizing the Retraining with TensorBoard

The script includes TensorBoard summaries that make it easier to understand, debug, and optimize the retraining. For example, you can visualize the graph and statistics, such as how the weights or accuracy varied during training.

To launch TensorBoard, run this command during or after retraining:

```bash
tensorboard --logdir /tmp/retrain_logs
```

Once TensorBoard is running, navigate your web browser to `localhost:6006` to view the TensorBoard.

The script will log TensorBoard summaries to `/tmp/retrain_logs` by default. You can change the directory with the `--summaries_dir` flag.

lp the network learn to cope with all the distortions that will occur in real-life uses of the classifier. The biggest disadvantage of enabling these distortions in our script is that the bottleneck caching is no longer useful, since input images are never reused exactly. This means the training process takes a lot longer, so I recommend trying this as a way of fine-tuning your model once you've got one that you're reasonably happy with.

You enable these distortions by passing `--random_crop`, `--random_scale` and `--random_brightness` to the script. These are all percentage values that control how much of each of the distortions is applied to each image. It's reasonable to start with values of 5 or 10 for each of them and then experiment to see which of them help with your application. `--flip_left_right` will randomly mirror half of the images horizontally, which makes sense as long as those inversions are likely to happen in your application. For example it wouldn't be a good idea if you were trying to recognize letters, since flipping them destroys their meaning.



For more information:

https://www.tensorflow.org/tutorials/image_retraining
