The policy training data provided below are recordings of episodes (runs) of
humans teleoperating robots in simulation. The data set includes hundreds of
episodes for each of the skills evaluated in [LBM Eval 1.0](README.md).

# Files

The dataset is composed of the following files:
- BimanualHangMugsOnMugHolderFromDryingRack.tar
- BimanualHangMugsOnMugHolderFromTable.tar
- BimanualLayCerealBoxOnCuttingBoardFromTopShelf.tar
- BimanualLayCerealBoxOnCuttingBoardFromUnderShelf.tar
- BimanualPlaceAppleFromBowlIntoBin.tar
- BimanualPlaceAppleFromBowlOnCuttingBoard.tar
- BimanualPlaceAvocadoFromBowlIntoBin.tar
- BimanualPlaceAvocadoFromBowlOnCuttingBoard.tar
- BimanualPlaceFruitFromBowlIntoBin.tar
- BimanualPlaceFruitFromBowlOnCuttingBoard.tar
- BimanualPlacePearFromBowlIntoBin.tar
- BimanualPlacePearFromBowlOnCuttingBoard.tar
- BimanualPutMugsOnPlatesFromDryingRack.tar
- BimanualPutMugsOnPlatesFromTable.tar
- BimanualPutRedBellPepperInBin.tar
- BimanualPutSpatulaOnPlateFromDryingRack.tar
- BimanualPutSpatulaOnPlateFromTable.tar
- BimanualPutSpatulaOnPlateFromUtensilCrock.tar
- BimanualPutSpatulaOnTableFromDryingRack.tar
- BimanualPutSpatulaOnTableFromUtensilCrock.tar
- BimanualStackPlatesOnTableFromDryingRack.tar
- BimanualStackPlatesOnTableFromTable.tar
- BimanualStoreCerealBoxUnderShelf.tar
- DumpVegetablesFromSmallToLargeContainer.tar
- PickAndPlaceBox.tar
- PlaceCupByCoaster.tar
- PlaceCupOnCoaster.tar
- PushCoasterToCenterOfTable.tar
- PushCoasterToMug.tar
- PutBananaInCenterOfTable.tar
- PutBananaOnSaucer.tar
- PutContainersOnPlate.tar
- PutCupInCenterOfTable.tar
- PutCupOnSaucer.tar
- PutFruitInLargeContainerAndCoverWithPlate.tar
- PutGreenAppleInCenterOfTable.tar
- PutGreenAppleOnSaucer.tar
- PutKiwiInCenterOfTable.tar
- PutKiwiOnSaucer.tar
- PutMugInCenterOfTable.tar
- PutMugOnSaucer.tar
- PutOrangeInCenterOfTable.tar
- PutOrangeOnSaucer.tar
- PutSpatulaInUtensilCrock.tar
- PutSpatulaInUtensilCrockFromDryingRack.tar
- SeparateFruitsVegetablesIntoContainers.tar
- TurnCupUpsideDown.tar
- TurnLargeContainerUpsideDown.tar
- TurnMugRightsideUp.tar

The files are available for https download under this base URL:
<https://tri-ml-public.s3.amazonaws.com/datasets/lbm-eval-v1.1-sim-training-data/>

# Directory layout

The directory layout encodes skills, stations (simulated robot workstations),
date stamps, and episode numbers. Some of the directory levels are not
significant for the structure of the public dataset, but are preserved to mirror
the TRI's internal layout as nearly as possible.

Each episode tarball has this directory structure inside:

    tasks/
      ${SKILL_NAME}/
        ${STATION}/
          sim/
            bc/
              teleop/
                ${DATE_STAMP}/
                  diffusion_spartan/
                    episode_${EPISODE_NUMBER}

with variables filled in as follows:

* SKILL_NAME: one of the skills known by LBM Eval, but spelled using CamelCase,
  rather than the snake_case used on the `evaluate` command line.
* STATION: one of `cabot` or `riverway`. All of the episodes of a given skill
  will use the same station.
* DATE_STAMP: ISO 8601 date format, with colons replaced by dashes, indicating
  the approximate date the episode was collected.
* EPISODE_NUMBER: a decimal number with no leading zeros.

# Diffusion spartan format

The spartan format is originally from Russ Tedrake's
[RobotLocomotion Group](https://groups.csail.mit.edu/locomotion/russt.html)
used in the
[Dense Object Nets (Dense Descriptor)](https://arxiv.org/abs/1806.08756) work.

The original doc can be found in their public repo:
[`RobotLocomotion/pytorch-dense-correspondence`](https://github.com/RobotLocomotion/pytorch-dense-correspondence/blob/76bf6499c325ad136a094fb341158a90eaa31d53/doc/data_organization.md#data-within-image-folders)

The "diffusion spartan" format builds on the original, adding more data, and
providing base data in different formats.

The time-sequence data (camera images, poses, etc.) are collected at 10Hz
during the simulation. Each such recording is called a "keyframe". A typical
episode contains hundreds of frames of data.

## Files in an episode

A typical episode subdirectory tree looks like this:

    processed/
      actions.npz
      detailed_reward_traj.npz
      detailed_task_predicate_traj.npz
      extrinsics.npz
      images_6CD146030E99/
        camera_info.yaml
        pose_data.yaml
      images_6CD146031C25/
        camera_info.yaml
        pose_data.yaml
      images_BFS_23595718/
        camera_info.yaml
        pose_data.yaml
      images_BFS_23595721/
        camera_info.yaml
        pose_data.yaml
      images_BFS_23595722/
        camera_info.yaml
        pose_data.yaml
      images_BFS_23595725/
        camera_info.yaml
        pose_data.yaml
      intrinsics.npz
      manifest.json
      metadata.yaml
      observations.npz
      resolved_scenario.yaml
      summary.npz

These file types are used:

* .json: [json format](https://en.wikipedia.org/wiki/JSON)
* .yaml: [yaml format](https://en.wikipedia.org/wiki/YAML)
* .npz: [npz data archive](https://numpy.org/devdocs/reference/generated/numpy.lib.npyio.NpzFile.html)

Here is a brief summary of file contents.

* manifest.json: a list of files in this episode, by s3 paths of the original
  (private) location. The mapping to public locations should be obvious.

* metadata.yaml: overall description of the episode. Importantly, the
  `camera_id_to_semantic_name` maps the camera ID strings ('6CD146030E99',
  'BFS_23595718', etc.) used throughout the data to human-readable names.
* resolved_scenario.yaml: detailed configuration of the simulation scenario;
  some of this is not interpretable with just the open-source LBM Eval software.
* images_*/camera_info.yaml: camera intrinsics (see below for more).
* images_*/pose_data.yaml: camera extrinsics (see below for more).

* actions.npz: time sequence of commands to the robot (see below).
* intrinsics.npz: camera intrinsics for all cameras.
* extrinsics.npz: camera extrinsics for all frames for all cameras.
* observations.npz: a large collection of time-sequence data (see
  below). Includes all RGB, depth and label images, and robot poses and forces.
* summary.npz: overall results. Importantly, `episode_success` is a 1x1 boolean
  array.

## Redundancies

Some data categories are redundantly encoded in each episode.

### Camera intrinsics

Intrinsics (camera matrix) are included in both
images_${CAMERA}/camera_info.yaml, and in the intrinsics.npz archive. In
addition, camera_info.yaml contains the image dimensions in pixels.

```
tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed$ ipython3

...

In [1]: import numpy as np

In [2]: np.load('intrinsics.npz')
Out[2]: NpzFile 'intrinsics.npz' with keys: 6CD146030E99, BFS_23595725, BFS_23595721, BFS_23595718, 6CD146031C25...

In [3]: i = np.load('intrinsics.npz')

In [4]: i['6CD146030E99']
Out[4]:
array([[616.129,   0.   , 321.269],
       [  0.   , 615.758, 247.864],
       [  0.   ,   0.   ,   1.   ]], dtype=float32)

...

tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed$ cat images_6CD146030E99/camera_info.yaml
camera_matrix:
  cols: 3
  rows: 3
  data:
    - 616.1290283203125
    - 0.0
    - 321.2690124511719
    - 0.0
    - 615.7579956054688
    - 247.86399841308594
    - 0.0
    - 0.0
    - 1.0
  image_height: 480
  image_width: 640
```

### Camera extrinsics

Extrinsics (camera poses by keyframe) are provided in both
images_${CAMERA}/pose_data.yaml, and in the extrinsics.npz archive. In
addition, the yaml gives the keyframe timestamps in microseconds, and the
phantom names of .png files (not provided).

```
tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed$ ipython3

...

In [1]: import numpy as np

In [2]: np.load('extrinsics.npz')
Out[2]: NpzFile 'extrinsics.npz' with keys: 6CD146030E99, BFS_23595725, BFS_23595721, BFS_23595718, 6CD146031C25...

In [3]: e = np.load('extrinsics.npz')

In [4]: e['6CD146030E99'].shape
Out[4]: (198, 4, 4)

In [5]: e['6CD146030E99'][0]
Out[5]:
array([[-0.09089504,  0.91194054, -0.40012817,  0.49251986],
       [ 0.98537069,  0.02419483, -0.16869859,  0.26256949],
       [-0.14416205, -0.40960843, -0.90079867,  0.90730754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

In [6]: e['6CD146030E99'][197]
Out[6]:
array([[-0.09089504,  0.91194054, -0.40012817,  0.49251986],
       [ 0.98537069,  0.02419483, -0.16869859,  0.26256949],
       [-0.14416205, -0.40960843, -0.90079867,  0.90730754],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

...

tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed$ head -n20 images_6CD146030E99/pose_data.yaml
0:
  camera_to_world:
    quaternion:
      w: 0.09014034753304856
      x: -0.6681520776522029
      y: -0.7099099479501487
      z: 0.20365506297925187
    translation:
      x: 0.4925198648813639
      y: 0.26256949292330367
      z: 0.9073075406104864
  depth_image_filename: 000000_depth.png
  rgb_image_filename: 000000_rgb.png
  timestamp: 0
1:
  camera_to_world:
    quaternion:
      w: 0.09014034753304856
      x: -0.6681520776522029
      y: -0.7099099479501487

```

## The actions archive (actions.npz)

The actions archive contains the skill type and actions (commanded robot poses)
for all keyframes.

The actions are 20-element vectors that encode the gripper poses and gripper
opening widths, as follows:

[ Rxyz | R_rot6d | Lxyz | L_rot6d | RG | LG ]

where:
* Rxyz - right gripper X,Y,Z translation in meters
* R_rot6d - right gripper rotation in 6d representation (see below)
* Lxyz - left gripper X,Y,Z translation in meters
* L_rot6d - left gripper rotation in 6d representation (see below)
* RG - right gripper opening in meters
* LG - left gripper opening in meters

The 6d rotation representation is a truncated and flattened 3x3 rotation
matrix. See [On the Continuity of Rotation Representations in Neural
Networks, 2019](http://arxiv.org/abs/1812.07035).

The skill type records are both redundant and difficult to decode (stored via
python `pickle`). At each keyframe, a pickled-object encoding of the skill type
name is repeated. It will be the same name encoded in the directory structure
for the episode.

For completeness' sake, here is a method of fully decoding the actions
archive. Note that it is performed within a virtual environment containing the
LBM Eval 1.0 software and with `ipython` installed. The exact sequence of import
statements will load the necessary `SkillType` definition to interpret the
`skill_type` data:

```
(venv) ~/tmp/venv$ ipython

...

In [1]: from lbm_eval.evaluate import main

In [2]: from anzu.intuitive.skill_defines import SkillType

In [3]: import numpy as np

In [4]: a = np.load('tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed/actions.npz', allow_pickle=True)

In [5]: dir(a.f)
Out[5]: ['actions', 'skill_type']

In [6]: a.f.actions.shape
Out[6]: (198, 20)

In [7]: a.f.actions[0]
Out[7]: 
array([-0.28634044,  0.2481908 ,  0.43399996,  0.66238399,  0.74652414,
        0.06284232,  0.74852294, -0.66295501, -0.01428483, -0.25651794,
       -0.31283726,  0.44475016,  0.84605298, -0.5275624 ,  0.07663069,
       -0.52940845, -0.84835496,  0.00453379,  0.1       ,  0.1       ])

In [8]: a.f.skill_type.shape
Out[8]: (198,)

In [9]: a.f.skill_type[0]
Out[9]: <SkillType.PlaceCupByCoaster: 'place_cup_by_coaster'>
```

## The observations archive (observations.npz)

The observations archive contains camera image data for all channels of all
cameras, plus traces of the robot low-level control channels, desired and actual.

```
tasks/PlaceCupByCoaster/cabot/sim/bc/teleop/2024-11-18T15-43-21-05-00/diffusion_spartan/episode_42/processed$ ipython3

...

In [1]: import numpy as np

In [2]: o = np.load('observations.npz')

In [3]: dir(o.f)
Out[3]:
['6CD146030E99',
 '6CD146030E99_depth',
 '6CD146030E99_label',
 '6CD146031C25',
 '6CD146031C25_depth',
 '6CD146031C25_label',
 'BFS_23595718',
 'BFS_23595718_depth',
 'BFS_23595718_label',
 'BFS_23595721',
 'BFS_23595721_depth',
 'BFS_23595721_label',
 'BFS_23595722',
 'BFS_23595722_depth',
 'BFS_23595722_label',
 'BFS_23595725',
 'BFS_23595725_depth',
 'BFS_23595725_label',
 'robot__actual__external_wrench__left::panda',
 'robot__actual__external_wrench__right::panda',
 'robot__actual__grippers__left::panda_hand',
 'robot__actual__grippers__right::panda_hand',
 'robot__actual__joint_position__left::panda',
 'robot__actual__joint_position__right::panda',
 'robot__actual__joint_torque__left::panda',
 'robot__actual__joint_torque__right::panda',
 'robot__actual__joint_torque_external__left::panda',
 'robot__actual__joint_torque_external__right::panda',
 'robot__actual__joint_velocity__left::panda',
 'robot__actual__joint_velocity__right::panda',
 'robot__actual__poses__left::panda__rot_6d',
 'robot__actual__poses__left::panda__xyz',
 'robot__actual__poses__right::panda__rot_6d',
 'robot__actual__poses__right::panda__xyz',
 'robot__actual__wrench__left::panda',
 'robot__actual__wrench__right::panda',
 'robot__desired__external_wrench__left::panda',
 'robot__desired__external_wrench__right::panda',
 'robot__desired__grippers__left::panda_hand',
 'robot__desired__grippers__right::panda_hand',
 'robot__desired__joint_position__left::panda',
 'robot__desired__joint_position__right::panda',
 'robot__desired__joint_torque__left::panda',
 'robot__desired__joint_torque__right::panda',
 'robot__desired__joint_torque_external__left::panda',
 'robot__desired__joint_torque_external__right::panda',
 'robot__desired__joint_velocity__left::panda',
 'robot__desired__joint_velocity__right::panda',
 'robot__desired__poses__left::panda__rot_6d',
 'robot__desired__poses__left::panda__xyz',
 'robot__desired__poses__right::panda__rot_6d',
 'robot__desired__poses__right::panda__xyz',
 'robot__desired__wrench__left::panda',
 'robot__desired__wrench__right::panda',
 'robot__version']
```
