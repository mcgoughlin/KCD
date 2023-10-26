1) Tried just plain training of 2D axial tile and 3D axial patch model on coreg_ncct dataset. Got cross-val AUC of 0.845 / 0.81 respectively. Tested both model on unseen data, got sensitivity of about 25%, whereas 3D model got 45%. We believe this is due to the integration technique of converting many slice-wise inferences into a kidney-wise prediction; 3D model is simply top-1, whereas we employed top-10 with 2D. We should test 2D in top-1, to ensure this comparison is fair and valid. 2D algorithm used pretraining ResNeXT weights, 3D was randomly initialised.
2) Given that the 2D model had some pretraining, we wanted to give the 3D model pretraining too. Thus, we ran it through kits23 and kits23sncct data before finetuning. We observed that the 3D model's cross-val performance increased to 0.84AUC.  I should do this pretraining on 2D classifier, to see if it improves its accuracy, too.
3) Then wanted to evaluate the effect of multi-task learning. We coded up a decoder branch to ResNeXT and trained it via segmentation and classification on the Kits23 dataset, before regular classification training on coreg_ncct. We observed that the cross-val performance of the 3D model increased to 0.86AUC. We should do this on the 2D model, too.

Tasks:

- [ ] Train 2D model in top-1 mode
- [ ] Train 2D model on coreg_ncct dataset after pretraining on kits23
- [ ] Train 2D model with decoder branch on kits23 MTL

After these tasks are complete, I should investigate further MTL improvements via CECT synthesis/prediction, based on this paper: https://www.medrxiv.org/content/10.1101/2022.01.12.22269120v1 
