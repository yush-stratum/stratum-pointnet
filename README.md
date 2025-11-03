# Geotech Discontinuity Detection using Point Clouds

I am using this markdown to document my understanding of every component at play here.

# Model Training

## 1. Data Preparation

Since we plan on training/testing on the same dataset for a while, we typically save them on disk and read at runtime instead of re-generating the same data all over again.

We can look into `prepare_data_for_training` and underline its multiple responsibilities:
- Reads in point cloud used for training/test and loads the label DXFs
- The DXFs are all Line entities but they all form closed polygons. `lines_to_polygons` handles an array of lines and returns a list of lists where each element is a polygon. A polygon is here represented by a series of joints ( intersection of two lines )
- `label_points_in_polygon_2d_projection` is a key operation. It takes in each polygon and the point cloud and returns an array where each point has been labelled if it falls within a polygon.
- `create_train_test_split` takes two points, casts them into a line and everything to the left of the line is training data, everything to the right is test data. This is mid and should be changed whenever we re-engineer the pipeline.
- `filter_polygons_by_mask` filters the binary labels we have into train/test splits.
- Note that the train mask and test mask are verified and are undoubtedly correct.

## 2. Creating Datasets
- We will closely examine `RockJointDataset` here as it is the key part of this step.
- RockJointDataset is instantiated with a 3D positional array, a 1D label array, a dictionary of polygons where each key is the label, patch_size to regulate the size of patches extracted, normalize_mode allows the user to choose multiple ways to normalize data, an an augment boolean to supplement the training pipeline. Upon initialization we call `_extract_patches` which makes patches and pointwise labels.
- Algorithm steps:
    - We estimate the average point spacing by taking a random sample of at most 10k points, for each of these points we find their 50 closest points. We now have 10k x 50 points, we take the mean of those to get the average point spacing.

    - Point spacing is equivalent to density, we use that to then calculate our patch radius such that each query will contain at most `patch_size` points.

    - We sample at most 5k points which have labels. We use two metrics that guide our selection heuristic: `min_coherence` and `min_labeled_ratio`. `min_coherence` is the least percentage of labelled points that match the center label. `min_labeled_ratio` is the proportion of points that have labels since sometimes we have a lot of -1 labels.

    - For each sampled center:
        - Query a ball of radius `patch_radius`.
        - If labelled ratio < min label ratio, reject. If more than X% of pts dont match the center label, reject.
        - If patch has more points than patch_size, we sample the points. If less, we pad by repeating random points. (Note: This could be a red flag, why are we padding with "random" points, that makes us lose spatial coherence. The only good thing here is that these patches already pass the "coherence" and "min_labelled" filters, so maybe its not that bad. Orange flag for sure.)

- The rest of the function is descriptive statistics about the patches.

- normalize_mode has center, center_scale and None. center just translates all the points wrt mean(XYZ in patch). center_scale translates and scales the point given the mean and std dev.

- augment appplies a rotation, random scaling, jitter and random dropout to a given patch. We dont currently use this, i think its bad.


- Let's look at `RockJointInferenceDataset`. Since we do not have polygons at inference time, we have to use another strategy to pass patches in. Here we call `_create_spatial_grid`.
- We sample 10k points to find the density, which we then use to get the spatial stride.
- NOTE: Not sure why avg_spacing * sqrt(self.stride) is used here. My stride is 256, sqrt(256) is 16, if the avg spacing is 0.5m, the query space is 8m. At train time the patch radius is 3.3m . We can set the stride in config to be 64, then the query space might be 4m just like training. Need to be fixed to add more consistency between train and test.
- Given a stride and min/max bounds on xyz, we create a 3d linspace.
- Now each voxel in our grid has a centroid, we query `spatial_stride` x 1.5
- NOTE: Why do we even scale the fucking stride by 50% here? So now we are querying at 12m... Absolute tomfoolery. I removed it.
- the `getitem` method seems to sample random points if patch_points > patch_size, and pads by adding random points from patch_points.
- NOTE: Not sure im liking a lot of this random selection.

- We instantiate the train and test dataloaders. We calculate label balance, nothing crazy.

## 3. Model Creation
-  We implement PointNet++ Segmentation to carry out pointwise classification.

### PointNetSetAbstraction

- Furthest Point Sampling
    - Algorithm:
        - For each point in npoints ( number of points chosen to be sampled):
            - We sample 1 point per batch.
            - Calculate the distance from all points to the sampled points.
            - Only update the distance array when the distance from centroid to PCL is less than prior.
            - We take the index of the furthest point per batch and update the centroids array. centroids array holds points that are furthest than their successors.
    - Let's say we randomly sample x1, we sample x2 which is the furthest point from x1. Now we need to sample x3 which is furthest from both x1,x2. Now we need to sample x4 which is furthest from x1,x2,x3. It goes on like this.

- Index Points: Batch-aware indexing. Makes sure you get the right points from the right batch.

- Query Ball Point: For each new point in new_xyz, find nsample points within self.radius from xyz

- What do we have now: For each furthest point, we sample nsample closest points within a radius. That provides more support to the maximal set. These points are normalized. We have two outputs. `new_xyz` and `new_points`
- `new_xyz`: npoint from FPS
- `new_points`: points queried around FPS with ball.

- How do i interpret the PSA constructor and the forward call?
Let's take an example of `self.sa1 = PointNetSetAbstraction(
          npoint=512, radius=0.2, nsample=32,
          in_channel=input_channels, mlp=[32, 32, 64], group_all=False)` and self.sa1(xyz,None) where xyz here is every point in a patch in 3D.


self.sa1(xyz,None) would run FPS on 512 points. For each point, we'd sample 32 points within 20cm. we now pass these points in an MLP->BN->ReLU. We then take the maximum feature value across all nsample neighbours for each feature channel. We do this 4 times with different radii and different number of points for FPS.

## PointNetFeaturePropagation

## 4. Model Training
- Undertakes training and validation epochs as well as metric tracking.
- Training Epoch:
  - pass the patches into the model
  - calculate loss on points where we have labels
  - step


# Model Inference
We will go thru `inference.py`. The main class is `PointCloudInference` and it implements a `predict_point_cloud` method which carries out the inference, the other methods are util to load and save.
- `predict_point_cloud` iterates over the infernce dataloader. We went over `RockJointInferenceDataset` above which creates a grid which we will iterate over.
- We pass the batch thru the model, it outputs the probs, we take argmax to get the class idx
- We add the probabilities in an array and track the number of times this point was predicted so we can later on take the mean prediction at that point.


## Issues:
- WE ARE NOT DOING MULTI SCALE GROUPING. But i doubt this would fix it.
- When passing input dim in model = 6, it defaults to 3 for some reason. Seems to be having this issue since we moved to using voxels


- Training data cannot be reproduced by learning patches of points in XYZ. I visited the patch logic and it seems sound from a high level. The only issues are thigns I mentionned above, the random sampling could be a problem.

## Thigns to try:
- 3 classes. Background, Joint, No Joint.
- MSG
- XYZ + RGB + Normals ( Combinations of 2 or all 3 )

### Voxel Dataset questions
- If we are making a grid and then picking based on min_pts, what happens when the number of points in the voxel are greater than the patch_size?

- At inference time, do we predict on every point? If so, how do we predict on a voxel that has more than patch_size points.
