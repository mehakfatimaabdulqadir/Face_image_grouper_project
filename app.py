import csv
import hashlib
import io
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# -------------------- Page setup --------------------

st.set_page_config(
    page_title="Face Image Grouper",
    page_icon="📷",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "results" / "facenet_vggface2_weights.pth"


# -------------------- Model loading --------------------

@st.cache_resource
def load_models():
    """Load models once and reuse them across app reruns."""
    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        selection_method="center_weighted_size",
        device="cpu"
    )

    if WEIGHTS_PATH.is_file():
        model = InceptionResnetV1(pretrained=None)
        weights = torch.load(
            WEIGHTS_PATH,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(weights)
    else:
        # Downloads weights on first use, then uses Torch's cache.
        model = InceptionResnetV1(pretrained="vggface2")

    model.eval()

    # Cached models are shared; serialize inference calls.
    return detector, model, threading.Lock()


# -------------------- Face processing --------------------

def process_images(uploads, detector, model, model_lock):
    """Extract one centre-weighted face and embedding per image."""
    vectors = []
    records = []
    skipped = []

    progress = st.progress(0)
    status = st.empty()

    try:
        with model_lock, torch.inference_mode():
            for index, upload in enumerate(uploads):
                status.write(
                    f"Processing {index + 1}/{len(uploads)}: "
                    f"{upload.name}"
                )

                try:
                    with Image.open(
                        io.BytesIO(upload.getvalue())
                    ) as image:
                        image = ImageOps.exif_transpose(image)
                        image = image.convert("RGB")
                        face = detector(image)

                    if face is None:
                        skipped.append({
                            "file": upload.name,
                            "reason": "No face detected"
                        })
                    else:
                        embedding = model(
                            face.unsqueeze(0)
                        ).cpu().numpy()[0]

                        # Reverse standardization for crop display.
                        crop = face.cpu().permute(1, 2, 0).numpy()
                        crop = (
                            (crop * 128 + 127.5) / 255
                        ).clip(0, 1)

                        vectors.append(embedding)
                        records.append({
                            "image_id": index + 1,
                            "filename": upload.name,
                            "crop": (crop * 255).astype(np.uint8)
                        })

                except (OSError, ValueError) as error:
                    skipped.append({
                        "file": upload.name,
                        "reason": str(error)
                    })

                progress.progress(
                    (index + 1) / len(uploads)
                )
    finally:
        progress.empty()
        status.empty()

    return vectors, records, skipped


def create_csv(records, labels):
    """Export image-to-cluster assignments."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["image_id", "filename", "cluster"])

    for record, label in zip(records, labels):
        writer.writerow([
            record["image_id"],
            record["filename"],
            int(label)
        ])

    return buffer.getvalue().encode("utf-8-sig")


# -------------------- Interface --------------------

st.title("📷 Face Image Grouper")
st.write(
    "Upload face photos, choose the number of groups, "
    "and view faces grouped by similarity."
)

with st.sidebar:
    st.header("Grouping settings")

    number_of_clusters = st.number_input(
        "Number of groups (K)",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )

    st.info(
        "Use multiple photos per person. When an image contains "
        "several faces, the app selects one using face size "
        "and distance from the image centre."
    )

    st.caption(
        "Method: MTCNN → FaceNet embeddings → "
        "L2 normalization → K-Means."
    )

uploads = st.file_uploader(
    "Upload JPG, JPEG or PNG images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.caption(
    "Group numbers are similarity labels, not predicted names. "
    "This app creates new groups from the current uploads."
)

# Identify current inputs so old results are not shown after changes.
signature_builder = hashlib.sha256(
    str(number_of_clusters).encode()
)

for upload in uploads:
    signature_builder.update(upload.name.encode())
    signature_builder.update(
        hashlib.sha256(upload.getvalue()).digest()
    )

signature = signature_builder.hexdigest()

if st.session_state.get("input_signature") != signature:
    st.session_state.pop("result", None)
    st.session_state["input_signature"] = signature

if uploads:
    st.write(f"**Uploaded images:** {len(uploads)}")

run_clicked = st.button(
    "Group Faces",
    type="primary",
    disabled=not uploads
)

if run_clicked:
    st.session_state.pop("result", None)

    if len(uploads) <= number_of_clusters:
        st.warning(
            "Upload more images than the selected number of groups."
        )
    else:
        try:
            with st.spinner("Loading face models..."):
                detector, model, model_lock = load_models()

            vectors, records, skipped = process_images(
                uploads, detector, model, model_lock
            )

            if len(vectors) <= number_of_clusters:
                st.error(
                    f"Only {len(vectors)} valid faces were found. "
                    "Add more photos or reduce K."
                )
                if skipped:
                    st.table(skipped)
            else:
                features = normalize(
                    np.stack(vectors),
                    norm="l2"
                )

                estimator = KMeans(
                    n_clusters=int(number_of_clusters),
                    init="k-means++",
                    n_init=20,
                    random_state=42
                )

                labels = estimator.fit_predict(features)
                found_groups = len(np.unique(labels))

                if found_groups != number_of_clusters:
                    st.error(
                        "The embeddings did not form the requested "
                        "number of distinct groups. Try fewer groups "
                        "or more varied images."
                    )
                else:
                    score = silhouette_score(features, labels)

                    st.session_state["result"] = {
                        "features": features,
                        "labels": labels,
                        "records": records,
                        "skipped": skipped,
                        "silhouette": float(score),
                        "k": int(number_of_clusters)
                    }

        except Exception as error:
            st.error(f"Processing failed: {error}")


# -------------------- Results --------------------

result = st.session_state.get("result")

if result is not None:
    labels = result["labels"]
    records = result["records"]

    st.success("Face grouping complete!")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Processed faces", len(records))
    col2.metric("Skipped images", len(result["skipped"]))
    col3.metric("Groups", result["k"])
    col4.metric(
        "Silhouette score",
        f"{result['silhouette']:.4f}"
    )

    st.caption(
        "Silhouette measures cluster separation; it is not "
        "identity accuracy. ARI requires ground-truth identity "
        "labels and is not calculated for unlabeled uploads."
    )

    if result["skipped"]:
        with st.expander("Skipped images and reasons"):
            st.table(result["skipped"])

    st.subheader("Cluster sizes")

    sizes = [
        int(np.sum(labels == cluster_id))
        for cluster_id in range(result["k"])
    ]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(
        [f"Cluster {i}" for i in range(result["k"])],
        sizes
    )
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("Number of faces")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Face embedding visualization")

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(result["features"])

    fig, ax = plt.subplots(figsize=(8, 5))

    for cluster_id in range(result["k"]):
        mask = labels == cluster_id
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.8
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Groups")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        f"2D PCA captures "
        f"{pca.explained_variance_ratio_.sum():.1%} of variance. "
        "Clustering uses all 512 features."
    )

    st.subheader("Grouped face crops")
    st.caption(
        "These are the actual face crops used for embeddings. "
        "Inspect them for incorrect face selection."
    )

    for cluster_id in range(result["k"]):
        indices = np.where(labels == cluster_id)[0]

        with st.expander(
            f"Cluster {cluster_id} — {len(indices)} faces",
            expanded=True
        ):
            columns = st.columns(5)

            for position, index in enumerate(indices):
                record = records[index]
                columns[position % 5].image(
                    record["crop"],
                    caption=(
                        f"#{record['image_id']} "
                        f"{record['filename']}"
                    ),
                    width=150
                )

    st.download_button(
        "Download cluster assignments CSV",
        data=create_csv(records, labels),
        file_name="face_cluster_assignments.csv",
        mime="text/csv"
    )