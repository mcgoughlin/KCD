*Automated Small Kidney Cancer Detection in Non-Contrast Computed Tomography*

[Link to paper](https://arxiv.org/abs/2312.05258)


## Future ideas ##

For segmentation-based detection:

[Toda et al](https://pubmed.ncbi.nlm.nih.gov/34935652/) use a volume thresholding, with an implicit confidence thresholding of 0.5.

[Pancreatic PANDA framework](https://www.nature.com/articles/s41591-023-02640-w) uses a multi-task CNN for segmentation and classification. They use a Dice loss for segmentation and a cross-entropy loss for classification. 
This allows them to generate an ROC based on the classification head's probability output.