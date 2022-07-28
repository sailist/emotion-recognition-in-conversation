"""
Used for pre-train

VGGFace: https://www.robots.ox.ac.uk/~vgg/data/vgg_face/
The dataset consists of 2,622 identities. Each identity has an associated text file containing URLs for images and corresponding face detections. Please read the licence file carefully before downloading the data. Models pretrained using this data can be found at VGG Face Descriptor webpage. Please check the MatConvNet package release on that page for more details on Face detection and cropping.

VGGFace2: https://academictorrents.com/details/535113b8395832f09121bc53ac85d7bc8ef6fa5b
- The dataset contains 3.31 million images of 9131 subjects (identities), with an average of 362.6 images for each subject. Images are downloaded from Google Image Search and have large variations in pose, age, illumination, ethnicity and profession (e.g. actors, athletes, politicians). The whole dataset is split to a training set (including 8631 identites) and a test set (including 500 identites).


VGGFace2-HQ: https://github.com/NNNNAI/VGGFace2-HQ
"""


def vggface(root, split='train'):
    pass


def vggface2(root, split='train'):
    pass


def vggface2hq(root, split='train'):
    pass
