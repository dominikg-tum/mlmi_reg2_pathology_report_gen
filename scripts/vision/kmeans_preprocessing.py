from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm


def compress_image(image, number_of_clusters, keep_kmeans_objects=False):
    """
    Compress image colors using KMeans clustering.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image with shape (H, W, 3), values in [0, 255].
    number_of_clusters : int
        Number of color clusters.
    keep_kmeans_objects : bool
        If True, return both compressed image and fitted KMeans object.

    Returns
    -------
    compressed_image : np.ndarray
        KMeans-compressed RGB image.
    km : KMeans, optional
        Fitted KMeans object.
    """

    # Reshape image to color space: H*W x 3
    img = image.reshape(-1, image.shape[2]) / 255.0

    # Apply KMeans
    km = KMeans(
        n_clusters=number_of_clusters,
        random_state=0,
        n_init="auto"
    )

    km.fit(img)
    labels = km.predict(img)

    # Replace each pixel by its cluster center
    new_colors = km.cluster_centers_[labels]

    # Convert back to image format
    compressed_image = (new_colors * 255).reshape(image.shape)

    # Clip values and convert to uint8
    compressed_image = np.clip(compressed_image, 0, 255).astype("uint8")

    if keep_kmeans_objects:
        return compressed_image, km
    else:
        return compressed_image


# Input/output folders
input_dir = Path("/mnt/projects/mlmi/reg2/dataset/thumbnails")
output_dir = Path("/mnt/projects/mlmi/reg2/dataset/thumbnails_kmeans_5")
output_dir.mkdir(parents=True, exist_ok=True)

# Number of clusters
number_of_clusters = 5

# Read all jpg/png images
image_files = sorted(
    list(input_dir.glob("*.jpg")) +
    list(input_dir.glob("*.jpeg")) +
    list(input_dir.glob("*.png"))
)

print(f"Found {len(image_files)} images")

for image_path in tqdm(image_files):
    try:
        # Read image as RGB
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # Compress image
        cimg = compress_image(img_np, number_of_clusters)

        # Save result
        out_path = output_dir / image_path.name
        Image.fromarray(cimg).save(out_path, quality=95)

    except Exception as e:
        print(f"Failed: {image_path}")
        print(e)

print(f"Finished. Results saved to: {output_dir}")