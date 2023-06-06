# Optical RL-Gym

## Language

1. <a href="#english">English</a>
2. <a href="french">Français</a>


<a href="#english"><h2>Original Message from Carlos Natalino, creator of the Optical RL Gym toolkit</h2></a>

[OpenAI Gym](https://gym.openai.com/) is the de-facto interface for reinforcement learning environments.
Optical RL-Gym builds on top of OpenAI Gym's interfaces to create a set of environments that model optical network problems such as resource management and reconfiguration.
Optical RL-Gym can be used to quickly start experimenting with reinforcement learning in optical network problems.
Later, you can use the pre-defined environments to create more specific environments for your particular use case.

<a href="#bib">Please use the following bibtex:</a>

```
@inproceedings{optical-rl-gym,
  title = {The {Optical RL-Gym}: an open-source toolkit for applying reinforcement learning in optical networks},
  author = {Carlos Natalino and Paolo Monti},
  booktitle = {International Conference on Transparent Optical Networks (ICTON)},
  year = {2020},
  location = {Bari, Italy},
  month = {July},
  pages = {Mo.C1.1},
  doi = {10.1109/ICTON51198.2020.9203239},
  url = {https://github.com/carlosnatalino/optical-rl-gym}
}
```

## Features

Across all the environments, the following features are available:

- Use of [NetworkX](https://networkx.github.io/) for the topology graph representation, resource and path computation.
- Uniform and non-uniform traffic generation.
- Flag to let agents proactively reject requests or not.
- Appropriate random number generation with seed management providing reproducibility of results.

## Content of this document

1. <a href="#installation">Installation</a>
2. <a href="#environments">Environments</a>
3. <a href="#examples">Examples</a>
4. <a href="#resources">Resources</a>
5. <a href="#contributors">Contributors</a>
6. <a href="#contact">Contact</a>

<a href="#installation"><h2>Installation</h2></a>

You can install the Optical RL-Gym with:

```bash
git clone https://github.com/carlosnatalino/optical-rl-gym.git
cd optical-rl-gym
pip install -e .
``` 

You will be able to run the [examples](#examples) right away.

You can see the dependencies in the [setup.py](setup.py) file.

**To traing reinforcement learning agents, you must create or install reinforcement learning agents. Here are some of the libraries containing RL agents:**
- [Stable-baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorFlow Agents](https://www.tensorflow.org/agents)
- [ChainerRL](https://github.com/chainer/chainerrl)
- [OpenAI Baselines](https://github.com/openai/baselines) -- in maintenance mode

<a href="#environments"><h2>Environments</h2></a>

At this moment, the following environments are ready for use:

1. Optical Network
2. RWAEnv
3. RMSAEnv
4. DeepRMSAEnv
5. RMSADPPEnv
6. RMSASBPPEnv
7. DeepRMSADPPEnv
8. DeepRMSASBPPEnv
9. DeepRMSADPPKSPEnv
10. DeepRMSASBPPKSPEnv
11. RMCSAEnv
12. QoSConstrainedRA

More environments will be added in the near future.

<a href="#examples"><h2>Examples</h2></a>

Training a RL agent for one of the Optical RL-Gym environments can be done with a few lines of code.

For instance, you can use a [Stable Baselines](https://github.com/hill-a/stable-baselines) agent trained for the RMSA environment:

```python
# define the parameters of the RMSA environment
env_args = dict(topology=topology, seed=10, allow_rejection=False, 
                load=50, episode_length=50)
# create the environment
env = gym.make('RMSA-v0', **env_args)
# create the agent
agent = PPO2(MlpPolicy, env)
# run 10k learning timesteps
agent.learn(total_timesteps=10000)
```

We provide a set of [examples](./examples).

<a href="#resources"><h2>Resources</h2></a>

- Introductory paper `The Optical RL-Gym: an open-source toolkit for applying reinforcement learning in optical networks` (paper and video to be published soon).
- [List of publications using Optical RL-Gym](./docs/PUBLICATIONS.md)
- [How to implement your own algorithm](./docs/Implementation.md)

<a href="#contributors"><h2>Contributors</h2></a>

Here is a list of people who have contributed to this project:

- Igor M. de Araújo [[GitHub](https://github.com/igormaraujo/)]
- Paolo Monti [[Personal page](https://www.chalmers.se/en/staff/Pages/Paolo-Monti.aspx)]

- Youssef Alaoui Mrani (University College London, 3rd Year Project, 2020)
- Ella Gupta (University College London, 3rd Year Project, 2021)

<a href="#contact"><h2>Contact</h2></a>

This project is based on the toolkit initially created by Carlos Natalino [[Twitter](https://twitter.com/NatalinoCarlos)], who can be contacted through carlos.natalino@chalmers.se.

It was modified by Raphaël Yana from University College London for the purpose of a 3rd Year Project.




## French introduction

<a href="#french">Introduction en Français</a>

Ce recueil constitue un ensemble de fichiers dans le cadre d'un projet de 3ème année à la University College London (UCL), basé à Londres, qui a fait l'objet d'une thèse de fin d'études.
La thèse ne fut pas publiée.

[OpenAI Gym](https://gym.openai.com/) est une interface de facto créée pour les environnements de Deep Reinforcement Learning.
Optical RL-Gym se construit au sommet de l'interface d'OpenAI Gym, pout créer un ensemble d'environnements modélisant les problèmes des réseaux optiques tels que la gestion des ressources et la reconfiguration.
Optical RL-Gym peut être utilisée pour commencer rapidement l'experimentation avec le Reinforcement Learning dans les problèmes de réseaux optiques.
Plus tard, vous pouvez utiliser les environnements prédéfinis pour créer des environnements plus spécifiques pour les cas particuliers. 

<a href="#bib">Lien bibtex pour la documentation du dossier initial OpticalRL Gym</a>

## Attributs

A travers les environnements, les atributs suivants sont disponibles:

- Utilisation de [NetworkX](https://networkx.github.io/) pour la représenation graphique, la ressource et les calculs des réseaux de topologies.
- Génération de traffic uniforme et non-uniforme.
- Donner aux agents la possibilité de rejeter une requête ou non.
- Génération de nombres aléatoires appropriés avec une gestion des 'seeds', permettant une reproduction des résultats.

## Contenu de ce document

1. <a href="#installationf">Installation</a>
2. <a href="#environmentsf">Environnements</a>
3. <a href="#examplesf">Exemples</a>
4. <a href="#resourcesf">Ressources</a>
5. <a href="#contributorsf">Contributions</a>
6. <a href="#contactf">Contact</a>

<a href="#installationf"><h2>Installation</h2></a>

Vous pouvez installer le recueil de documents initial Optical RL-Gym à partir de:

```bash
git clone https://github.com/carlosnatalino/optical-rl-gym.git
cd optical-rl-gym
pip install -e .
``` 

Vous pourrez executer des [exemples](#examples) immédiatement.

Vous pouvez consulter les dépendances logicielles dans le fichier [setup.py](setup.py).

**Pour entrainer les agents de Reinforcement Learning, vous devez installer ou créer ces agent. Voici quelques bibliothèques contenant des agents RL:**
- [Stable-baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorFlow Agents](https://www.tensorflow.org/agents)
- [ChainerRL](https://github.com/chainer/chainerrl)
- [OpenAI Baselines](https://github.com/openai/baselines) -- en mode maintenance

<a href="#environmentsf"><h2>Environnements</h2></a>

A cet instant présent, les environnements suivants sont prêts à être utilisés:

1. Optical Network
2. RWAEnv
3. RMSAEnv
4. DeepRMSAEnv
5. RMSADPPEnv
6. RMSASBPPEnv
7. DeepRMSADPPEnv
8. DeepRMSASBPPEnv
9. DeepRMSADPPKSPEnv
10. DeepRMSASBPPKSPEnv
11. RMCSAEnv
12. QoSConstrainedRA

Plus d'environnements peuvent être créés dans le futur.

<a href="#examplesf"><h2>Exemples</h2></a>

Entrainer un agent RL pour un des environnements de Optical RL-Gym peut être fait avec quelques lignes de codes.

Par exemple, vous pouvez utiliser un agent [Stable Baselines](https://github.com/hill-a/stable-baselines) entraîné pour l'environnement RMSA:

```python
# define the parameters of the RMSA environment
env_args = dict(topology=topology, seed=10, allow_rejection=False, 
                load=50, episode_length=50)
# create the environment
env = gym.make('RMSA-v0', **env_args)
# create the agent
agent = PPO2(MlpPolicy, env)
# run 10k learning timesteps
agent.learn(total_timesteps=10000)
```

Nous fournissons un ensemble d'[exemples](./examples).

<a href="#resourcesf"><h2>Ressources</h2></a>

- Document d'introduction `The Optical RL-Gym: an open-source toolkit for applying reinforcement learning in optical networks`.
- [Liste des publications utilisant Optical RL-Gym](./docs/PUBLICATIONS.md)
- [Comment implémenter votre propre algorithme](./docs/Implementation.md)

<a href="#contributorsf"><h2>Contributions</h2></a>

Voici une liste des personnes ayant contribuées à ce projet ou initiallement au recueil de fichier Optical RL-Gym:

- Igor M. de Araújo [[GitHub](https://github.com/igormaraujo/)]
- Paolo Monti [[Personal page](https://www.chalmers.se/en/staff/Pages/Paolo-Monti.aspx)]

- Youssef Alaoui Mrani (University College London, Projet de 3ème Année, 2020)
- Ella Gupta (University College London, Projet de 3ème Année, 2021)

<a href="#contactf"><h2>Contact</h2></a>

Ce projet est basé sur le recueil de fichier créé par Carlos Natalino [[Twitter](https://twitter.com/NatalinoCarlos)],qui peut être contacté à carlos.natalino@chalmers.se.

Il fut modifié par Raphaël Yana, étudiant à la University College London (UCL, Londres) dans le c 3rd Year Project.



