This is the official repository for our paper:
[Gait-level analysis of mouse open field behavior using deep learning-based pose estimation](https://doi.org/10.1101/2020.12.29.424780).

[Zenodo link for models](https://zenodo.org/record/6380163)

# Installation

Before starting make sure you have `python3` installed. This code has been developed and tested on `python 3.8.10`. The recommended approach to installing dependencies is to use a virtual like:

    python3 -m venv gait-venv
    source gait-venv/bin/activate

    # now switch to the gait repo dir and install requirements
    cd $GAIT_REPO_DIR
    pip3 install -r requirements.txt

After successful installation you should be able to perform gait analysis on pose files. As an example you might run the command like:

    python3 gengaitstats.py \
          --batch-file ~/my-experiment-batch.txt \
          --root-dir ~/my-experiment-pose-dir \
          --out-file ~/my-experiment-gait.h5

Note that in this example `~/my-experiment-batch.txt` should contain a simple newline-separated file containing video input paths. As an example the file might look like this:

    LL1-B2B/2019-06-25_SPD/WT335G12N4F40108M-8-PSY.avi
    LL1-B2B/2019-06-25_SPD/WT335G12N4F40109M-9-PSY.avi
    LL1-B2B/2019-06-25_SPD/WT335G12N4F40111F-11-PSY.avi
    LL1-B2B/2019-06-25_SPD/WT335G12N4F40112F-12-PSY.avi
    LL2-B2B/2019-06-25_SPD/WT335G12N4F40113F-13-PSY.avi

and we expect root dir to contain the corresponding pose files using our `_pose_est_v2.h5` file suffix convention. So in this example the pose files corresponding to the batch file would be:

    ~/my-experiment-pose-dir/LL1-B2B/2019-06-25_SPD/WT335G12N4F40109M-9-PSY_pose_est_v2.h5
    ~/my-experiment-pose-dir/LL1-B2B/2019-06-25_SPD/WT335G12N4F40108M-8-PSY_pose_est_v2.h5
    ~/my-experiment-pose-dir/LL1-B2B/2019-06-25_SPD/WT335G12N4F40111F-11-PSY_pose_est_v2.h5
    ~/my-experiment-pose-dir/LL1-B2B/2019-06-25_SPD/WT335G12N4F40112F-12-PSY_pose_est_v2.h5
    ~/my-experiment-pose-dir/LL2-B2B/2019-06-25_SPD/WT335G12N4F40113F-13-PSY_pose_est_v2.h5

Note that this script doesn't require that the video files be available since it is working directly on the pose data. For more help using the script you can run the script with the help argument:

    python3 gengaitstats.py --help

# Licensing

This code is released under MIT license.

The data produced in the associated paper used for training models are released on Zenodo under a Non-Commercial license.
