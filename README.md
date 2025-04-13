# Improving ML MAPF Policies with Heuristic Search

This repo contains the techniques of 3 papers that focus on improving learned one-step policies for MAPF using heuristic search.
1. [Improving Learnt Local MAPF Policies with Heuristic Search (ICAPS 2024)](https://arxiv.org/abs/2403.20300)
2. [Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF (ICRA 2025)](https://arthurjakobsson.github.io/ssil_mapf/)
3. [Real-Time LaCAM (in submission)](https://arxiv.org/abs/2504.06091)

In particular, this repo contains:
1. CS-PIBT and LaCAM (from first paper)
2. Simple Scalable Imitation Learning model "SSIL" (from second paper)
3. One-step version of Real-Time LaCAM (from third paper)

Ths repo shows to use CS-PIBT, LaCAM, and one-step version of Real-Time LaCAM with a learnt policy. This codebase does not provide a command-line way to switch out different models, but users only need to modify the `runNNOnState()` function in `main_pys.simulator.py` to try out their own models.

## Installation
To clone the repository, run:
```sh
git clone git@github.com:Rishi-V/ML-MAPF-with-Search.git
```

To install dependencies, run:
```sh
conda config --set channel_priority flexible
conda env create -f environment.yml
conda activate mlmapf
```
This creates a conda environment named `mlmapf` that you should use.

To download data assets, i.e., maps, scenes, and the pretrained ``SSIL` model
```sh 
bash download_assets.bash
```
Note: gdown in the above command might complain at some point due to data limits or permissions. This is not a permission issue but a data limit issue, just wait a few minutes and rerun the command.

## Running CS-Freeze, CS-PIBT, Real-Time-LaCAM
To run the provided pre-trained model from `SSIL` on a map, look at the `simulator.py` file. Here is an example command:
```sh
python -m main_pys.simulator --mapNpzFile=data/constant_npzs/all_maps.npz \
      --mapName=den312d --scenFile=data/mapf-scen-random/den312d-random-1.scen \
      --bdNpzFile=data/constant_npzs/bd_npzs/den312d_bds.npz \
      --modelPath=data/model/ssil_model.pt \
      --outputCSVFile=logs/results.csv \
      --outputPathsFile=logs/paths.npy \
      --maxSteps=1000 --seed=0 --useGPU=True \
      --agentNum=200 --shieldType=CS-Freeze
```
Replace the last `--shieldType=CS-Freeze` with `--shieldType=CS-PIBT` or `--shieldType=Real-Time-LaCAM` to try out different collision shields. You can also try doing K multi-step planning using regular LaCAM with `--shieldType=LaCAM --lacamLookahead=K` where `K` is a positive integer of your choice.

To visualize outputs, use:
```sh
python -m main_pys.visualize_path den312d logs/paths.npy --scenName=den312d-random-1.scen 
```

We also provide a sample script to compare `CS-Freeze`, `CS-PIBT`, and `Real-Time-LaCAM` on different number of agents on the same map. You can try it out by running the following command (note this will take a few hours to run):
```sh
python -m main_pys.run_mini_test --mapName=den312d
```
At the end, you should see a plot like this in your `logs` folder.
<div align="center">
      <img src="example_den312d.png?raw=true" alt="Effect of different collision shields on den312d" width="500">
</div>

## Development
I will monitor this repo and try my best to answer questions/issues, but I likely cannot make large changes. If you are interested in improving this repo then I am happy to merge pull requests or add you as a collaborator to the repo.

Lastly, in general I am happy to collaborate so feel free to reach out.

## Citation
If you use this repository in your research, please cite our work:

```bibtex
@article{veerapaneni2024improving_mapf_policies_with_search,
  title = {Improving Learnt Local MAPF Policies with Heuristic Search},
  volume = {34},
  url = {https://ojs.aaai.org/index.php/ICAPS/article/view/31522},
  doi = {10.1609/icaps.v34i1.31522},
  number = {1},
  journal = {International Conference on Automated Planning and Scheduling (ICAPS)},
  author = {Veerapaneni, Rishi and Wang, Qian and Ren, Kevin and Jakobsson, Arthur and Li, Jiaoyang and Likhachev, Maxim},
  year = {2024},
  pages = {597-606},
}

@article{veerapaneni2024work_smart_not_harder,
  title = {Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large Scale Imitation Learning for MAPF},
  author = {Veerapaneni, Rishi and Jakobsson, Arthur and Ren, Kevin and Kim, Samuel and Li, Jiaoyang and Likhachev, Maxim},
  year = {2024},
  journal = {arXiv preprint arxiv:2409.14491},
  eprint = {2409.14491},
  archiveprefix = {arXiv},
  primaryclass = {cs.MA},
}

@article{liang2025real_time_lacam,
  title = {Real-Time LaCAM},
  author = {Liang, Runzhe and Veerapaneni, Rishi and Harabor, Daniel and Li, Jiaoyang and Likhachev, Maxim},
  year = {2025},
  journal = {arXiv preprint arxiv:2504.06091},
  eprint = {2504.06091},
  archiveprefix = {arXiv},
  primaryclass = {cs.MA},
}
```
