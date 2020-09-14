transform=logit
num_bootstrap_samples=1000
option=

python download_data.py
python paper_hyp_robust_imagenetv2.py --skip_download --transform=$transform # Figure 1 left
python paper_imagenetv2_bare.py --skip_download --transform=$transform # Figure 1 right
python paper_nat_shift_imagenetv2.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 2 top left
python paper_objectnet.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 2 top right
python paper_vid_robust_benign.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 2 bottom left
python paper_ytbb_robust_benign.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 2 bottom right
python paper_vid_robust_pmk.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 3 left
python paper_ytbb_robust_pmk.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 3 right
python paper_imagenet_a.py --skip_download --transform=$transform --num_bootstrap_samples=$num_bootstrap_samples --option=$option # Figure 4
python paper_eff_robust_corruptions.py --skip_download --transform=$transform # Figure 5 top row
python paper_eff_robust_pgd.py --skip_download --transform=$transform # Figure 5 bottom row
