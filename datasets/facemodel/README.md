# Introduction

In this document we describe how to replicate our high-resolution face
database. It takes a very long time to build the database. I recommend testing
the build process on a small database of 1000 images before you build the full database.

## Image Acquisition

There are four data sources. For each source we describe below how to
acquire the images. All images should go in the directory
`$(DFI_ROOT)/images/facemodel`.
Cite the relevant papers if you use these sources
in a publication.

* **CelebA**: Download the in-the-wild high-res images from
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. Look in filelist.txt
for the 31043 images we used.
The paths of the files should look like `$(DFI_ROOT)/images/facemodel/celeba/151094.png`.

* **Helen**: Download the images from
http://www.ifp.illinois.edu/~vuongle2/helen/. Look in filelist.txt for
the 1853 images we used.
The paths of the files should look like `$(DFI_ROOT)/images/facemodel/helen/helen_1/1012675629_1.png`.

* **MegaFace**: Register then download identities_0.tar.gz training images
from http://megaface.cs.washington.edu/. Look in filelist.txt for the
50000 images we used.
The paths of the files should look like
`$(DFI_ROOT)/images/facemodel/megaface/identities_0/100021856@N08_identity_16/9476029924_1.jpg`.

* **LFW-Google**: These are Internet photos found by searching with Google
Images. Note: the directory names are not the identities --- they are
the search terms. Look in lfwgoogle_url.txt for the 49868 original URLs.
The paths of the files should look like
`$(DFI_ROOT)/images/facemodel/lfwgoogle/Zydrunas_Ilgauskas/01042123.jpg`.

### (Optional) Image Cropping

Loosely cropping the images to the face will reduce I/O and save time.

## Image Preprocessing and Attribute detection
After all images are in `$(DFI_ROOT)/images/facemodel`, run
```python
  python database_rebuild.py
```
This ensures that all images contain faces, and detects visual attributes
(e.g. hair color, ethnicity, etc.) for all images. This script will take a
very long time (hours).

At the end of the first run you will be prompted to remove bad images
and duplicate images. Answer "yes" both times. If you deleted any images then
you will need to rerun database_rebuild.py (the second run should take less
than a minute). Deleting the images removes them from your hard disk. If you
do not want to delete them then you should manually move them outside the
`$(DFI_ROOT)/images/facemodel/` hierarchy and rerun database_rebuild.py.

