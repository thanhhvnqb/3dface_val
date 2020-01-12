
# HGFPN: 3D Facial Landmarks Detection for Intelligent Video Systems

## Prerequisite
- MXNet>=1.2.1
- tqdm==4.19.1
- Matplotlib
- NumPy
- scipy
- torchfile

## Repository structure
.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html

## Evaluation
- Run get_preds.py to get the predicted result of the proposed HGFPN
- Run evaluation_all.ipynb to get the AUC, NME scores.
- Run eval_aflw2000.ipynb to get the detailed score of AFLW2000(-Reannotated)
- Run Visualization to get the AUC curves
