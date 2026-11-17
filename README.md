<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/flow-dark.gif">
  <source media="(prefers-color-scheme: light)" srcset="assets/flow-light.gif">
  <img alt="Vortex shedding behind a slotted elliptical cylinder" src="assets/flow-light.gif" width="100%">
</picture>

<h1 align="center">Shyaam Mahesh</h1>

<p align="center">
  <b>Flows &nbsp;·&nbsp; Neural surrogates &nbsp;·&nbsp; Robots that have to trust them</b><br>
  M.Tech, Robotics &amp; Autonomous Systems, IISc Bengaluru &nbsp;|&nbsp; B.E. Mechanical, BITS Pilani
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/shyaam-mahesh"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://arxiv.org/abs/2610.33906"><img src="https://img.shields.io/badge/arXiv-2610.33906-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Physics%20of%20Fluids-38%2C%20073110-1f6feb?style=flat-square" alt="Physics of Fluids">
</p>

<br>

I started out running CFD and checking every number against experiment. Then I started training neural networks on that data, and the question changed: can a model learn a physical system well enough that you stop simulating and trust the model instead? These days I ask the robotics version of it, **how a robot should plan and act with learned models when it cannot know in advance where their predictions will fail.**

<br>

## Featured work

<table>
<tr>
<td width="50%" valign="top">
<a href="https://github.com/IDZTPHKKKS/go2-fall-risk"><img src="https://raw.githubusercontent.com/IDZTPHKKKS/go2-fall-risk/main/paper/figs/fork.gif" width="100%" alt="One Go2 state, two futures"></a>
<h3><a href="https://github.com/IDZTPHKKKS/go2-fall-risk">Will It Fall?</a></h3>
Do diffusion and flow-matching world models know how often a robot falls? I fork a Unitree Go2 simulation 128 ways at each state to get the true answer. They underestimate the risk, and a coarse-to-fine model fixes most of it.
<br><br>
<code>PyTorch</code> <code>MuJoCo</code> <code>world models</code>
</td>
<td width="50%" valign="top">
<a href="https://github.com/IDZTPHKKKS/Sybil4Drones"><img src="https://raw.githubusercontent.com/IDZTPHKKKS/Sybil4Drones/main/media/cave_demo.gif" width="100%" alt="Multi-drone exploration of a cave"></a>
<h3><a href="https://github.com/IDZTPHKKKS/Sybil4Drones">Sybil4Drones</a></h3>
One compromised drone creates fake teammates that claim frontiers and send fabricated scans. The defences use only each drone's own sensors. Tested in 2,350 missions on 70 maps. <a href="https://arxiv.org/abs/2610.33906">arXiv</a>
<br><br>
<code>multi-robot</code> <code>3D mapping</code> <code>security</code>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="https://github.com/IDZTPHKKKS/cfdml"><img src="https://raw.githubusercontent.com/IDZTPHKKKS/cfdml/main/docs/images/wakes.png" width="100%" alt="Bluff-body wakes"></a>
<h3><a href="https://github.com/IDZTPHKKKS/cfdml">cfdml</a></h3>
Command-line tools for CFD results: spectra, POD, DMD, grid convergence, Gaussian-process surrogates, and checks for long learned rollouts. Works on OpenFOAM cases, CSV and VTK.
<br><br>
<code>OpenFOAM</code> <code>POD / DMD</code> <code>surrogates</code>
</td>
<td width="50%" valign="top">
<h3><a href="https://github.com/IDZTPHKKKS/reynolds-grid-study">Reynolds × grid study</a></h3>
Over 200 FNO, U-Net and CNN surrogates trained on forced 2D Kolmogorov turbulence, each tested on every other Reynolds number and grid. The results were lopsided:
<br><br>
<table>
<tr><td>Re 32× higher than training</td><td>✅ holds up</td></tr>
<tr><td>Re 8× lower than training</td><td>❌ worse than "nothing changes"</td></tr>
</table>
The models don't remove energy fast enough, and telling them the viscosity doesn't fix it.
<br><br>
<code>FNO</code> <code>U-Net</code> <code>turbulence</code>
</td>
</tr>
</table>

## Papers

- **M. Shyaam**, P. Vishnuraj, K. K. Vijesh, S. Vengadesan. *Qualitative and quantitative analysis of wake dynamics in slotted elliptical cylinders for non-Newtonian fluids.* **Physics of Fluids** 38, 073110 (2026).
- **S. Mahesh**, A. B. Prakash, R. Sagwal, A. Gautam. *Sybil-Resilient Multi-Drone Exploration of Indoor and Confined Spaces.* [arXiv:2610.33906](https://arxiv.org/abs/2610.33906) (2026).
- S. Bhattacharyya, B. Souayeh, **S. Mahesh**, S. Khatri, N. K. Almulhem, M. W. Alam. *Vibration-assisted thermal performance enhancement of rectangular cooling channels using Fe₃O₄–TiO₂ hybrid nanofluid for EV battery cooling.* Accepted, **International Journal of Modern Physics B** (2026).

## Toolbox

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++">
  <img src="https://img.shields.io/badge/OpenFOAM-0B4F8A?style=flat-square" alt="OpenFOAM">
  <img src="https://img.shields.io/badge/ANSYS%20Fluent-FFB71B?style=flat-square&logo=ansys&logoColor=black" alt="ANSYS Fluent">
  <img src="https://img.shields.io/badge/ROS%202-22314E?style=flat-square&logo=ros&logoColor=white" alt="ROS 2">
  <img src="https://img.shields.io/badge/MuJoCo-1E1E1E?style=flat-square" alt="MuJoCo">
  <img src="https://img.shields.io/badge/Isaac%20Lab-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Isaac Lab">
  <img src="https://img.shields.io/badge/MATLAB-E16737?style=flat-square" alt="MATLAB">
  <img src="https://img.shields.io/badge/LaTeX-008080?style=flat-square&logo=latex&logoColor=white" alt="LaTeX">
</p>

<br>

<sub>The header is a lattice-Boltzmann simulation of flow past a slotted elliptical cylinder, the shape from my first paper. It is rendered by <a href="assets/lbm_banner.py"><code>assets/lbm_banner.py</code></a>. Away from the screen I play chess (FIDE 1708).</sub>
