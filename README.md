# Face Image Grouper

An unsupervised face image grouping project using pretrained
FaceNet embeddings, K-Means clustering, and a Streamlit interface.

## Method

1. Detect and crop one face per image using MTCNN.
2. Prefer large faces close to the image centre.
3. Extract 512-dimensional embeddings using InceptionResnetV1
   pretrained on VGGFace2.
4. Apply L2 normalization.
5. Group embeddings using K-Means.

Identity labels are not supplied to K-Means. Dataset identities
are used only for notebook evaluation.

## App Features

- Upload multiple JPG, JPEG, or PNG images.
- Set the number of groups.
- View grouped face crops and cluster sizes.
- View a PCA visualization and silhouette score.
- Download image-to-cluster assignments as CSV.

The app creates new clusters from the current uploads.
Cluster numbers are not predicted person names.

## Requirements

Python 3.11 is the tested environment.

## Local Setup — Windows CMD

Open CMD inside the project folder and run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

Pretrained FaceNet weights download automatically on first use
if local weights are unavailable. Internet is required for that
initial download.

## Dataset and Notebook

The experiments use a selected subset of Labeled Faces in the Wild
(LFW), using the funneled image archive.

- [Dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html)
- [Download image archive](https://ndownloader.figshare.com/files/5976015)

To reproduce the notebook:

1. Download and extract the archive.
2. Place the `lfw_funneled` folder beside the notebook.
3. Install Jupyter support:

   ```bat
   python -m pip install notebook ipykernel
   python -m ipykernel install --user --name facegrouper --display-name "Python (Face Grouper)"
   python -m notebook
   ```

4. Open the notebook and select `Python (Face Grouper)`.
5. Run the cells in order.

The notebook uses 100 development images from five identities,
followed by 31 additional images from four of those identities.
Dataset images and model binaries are excluded from this repository.

## Experimental Results

| Experiment | Images | ARI | Silhouette |
|---|---:|---:|---:|
| Largest-face selection baseline | 100 | 0.9746 | 0.4587 |
| Centre-weighted face selection | 100 | 1.0000 | 0.4678 |

Among the tested K values from 2 to 10, K=5 had the highest
silhouette score.

With the fitted cluster centres held fixed, the additional
31-image sample achieved:

- Fixed-mapping identity assignment accuracy: 100%
- Adjusted Rand Index: 1.0000
- Skipped images: 0

The cluster-to-identity mapping was derived from the development
sample, not from the additional-image answers.

## Limitations

- Results describe a small, selected dataset.
- Additional validation covers four identities and is imbalanced.
- Unseen identities were not evaluated.
- The face-selection rule was revised after inspecting a
  development-sample error.
- Images containing several faces may produce an incorrect crop.
- Blur, pose and lighting can affect embeddings.
- PCA shows only a two-dimensional projection.
- Silhouette and ARI are not classification accuracy percentages.
- Pretrained embedding models are used without training them
  from scratch.
