Breast Ultrasound Tumor Segmentation (YOLOv8)
=============================================

Overview
--------
This repository contains code and documentation for a breast ultrasound tumor segmentation project using a YOLOv8-based segmentation model. It includes a Jupyter notebook for experiments, a lightweight GUI for quick inference, and dependency information to reproduce results.

Author
------
Daud Sheh — CV Engineer

Repository contents
-------------------
- [breast-lesion-segmentation-yolov8.ipynb](breast-lesion-segmentation-yolov8.ipynb): Notebook used for dataset exploration, training experiments, and demonstration of training/inference pipelines with YOLOv8 segmentation.
- [gui.py](gui.py): Minimal GUI script to run inference on images using the trained YOLOv8 segmentation model. Launch with `python gui.py`.
- [requirements.txt](requirements.txt): Python package dependencies required to run the notebook and GUI (PyTorch, Ultralytics/YOLOv8, OpenCV, etc.).
- [commed-before-run.txt](commed-before-run.txt): Short notes / commands the author wanted to remember before running experiments. Check this file for quick pre-run tips.

Model & approach
-----------------
This project uses a YOLOv8 segmentation model (Ultralytics implementation). The model performs instance/semantic segmentation on breast ultrasound images to localize and segment suspicious lesions.

Key model notes
- Model architecture: YOLOv8 segmentation head trained on annotated ultrasound images.
- Weights: Not included in the repo by default — provide your trained weights as `weights/best.pt` (or update `gui.py` to point to your weights path).
- Training: Notebook contains the training loop and configuration (data splits, augmentation notes). Use the notebook to reproduce training runs or to adapt hyperparameters.

Setup
-----
1. Create and activate a Python 3.8+ virtual environment.
2. Install dependencies:

```
python -m pip install -r requirements.txt
```

3. Place model weights (if you have trained weights) at `weights/best.pt` or change the path used by `gui.py`.

Usage
-----
Notebook (experiments & training):
- Open [breast-lesion-segmentation-yolov8.ipynb](breast-lesion-segmentation-yolov8.ipynb) in Jupyter or Colab to run dataset preprocessing, model training, and evaluation cells. The notebook documents the steps used for training and inference.

GUI (quick inference):
- Run the GUI to load an image and perform segmentation using the model:

```
python gui.py
```

Notes about `gui.py`:
- The script expects a YOLOv8 segmentation model compatible with the Ultralytics API. If you changed the model API or saved a different format, update the loading code accordingly.

Reproducing training
--------------------
- Ensure dataset structure matches the notebook/data-loading cells (the notebook includes details on annotation format and directory layout).
- Use the notebook to run training cells; adjust dataset paths and hyperparameters as required.

Files explained (quick)
- [breast-lesion-segmentation-yolov8.ipynb](breast-lesion-segmentation-yolov8.ipynb): Full experiment record — preprocessing, dataloader, training, evaluation, visualizations.
- [gui.py](gui.py): Standalone script for inference and quick visualization.
- [requirements.txt](requirements.txt): Installable packages; keep versions pinned when reproducing results.
- [commed-before-run.txt](commed-before-run.txt): Workflow notes before executing experiments (read before running training/inference).

Tips for publishing to GitHub
---------------------------
- Add your trained weights to a release or provide a link to a cloud storage (do not commit large model files to the repo).
- Add sample images and a short demo GIF showing the GUI or notebook output.
- Create a `LICENSE` (MIT recommended) and an issue / contribution guide if you expect collaborators.

License & credits
-----------------
Add a license file (e.g., MIT) if you want permissive reuse. Credit the Ultralytics YOLOv8 implementation if you used their code.

Next steps I can help with
------------------------
- Prepare a GitHub-ready release (add a small demo, sample weights link, release notes).
- Update `gui.py` to load weights from a configurable path or add a CLI.
- Create a small `scripts/` folder with convenience commands for inference and evaluation.

Contact
-------
Daud Sheh — CV Engineer
