# neural-motion-graph

## Prepare the environment
```
conda env create --prefix "prefix of your choice" --file nmg_conda_env.yml
```

## Training

```
python src/nmg/scripts/ray_trainer.py
```
This will start multiple training sessions with hyperparameters option defined in the file `ray_trainer.py` and provided in the file `src/nmg/scripts/hparams.yaml`. The logs and the checkpoints will be available in the directory `nmg_logs`.

## Visualization
Edit the checkpoint path (ckpt_path) in the file `src/nmg/scripts/eval.py`.
```
python src/nmg/scripts/eval.py
```
This will save the output ".npz" file in the same directory as the checkpoint. Visualize the results by running the `src/nmg/scripts/viz_nmg.py` in the blender. For the reconstruction modes, the first half samples are the original and the rest half are the reconstructions.


