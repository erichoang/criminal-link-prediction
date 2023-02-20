train-cora-small:
	python train_inductive.py  --dataset cora --transductive_val --negative_sampling_ratio_train 1.2 --n_epochs 400

train-cora-small-ind-val:
	python train_inductive.py  --dataset cora --inductive_val --negative_sampling_ratio_train 1.2 --n_epochs 400

train-cora-full:
	python train_inductive.py  --dataset corafull --transductive_val --negative_sampling_ratio_train 1.2 --n_epochs 200

train-cora-full-ind-val:
	python train_inductive.py  --dataset corafull --inductive_val --negative_sampling_ratio_train 1.2 --n_epochs 200

train-citeseer:
	python train_inductive.py  --dataset citeseer --transductive_val --negative_sampling_ratio_train 3 --n_epochs 500

train-citeseer-ind-val:
	python train_inductive.py  --dataset citeseer --inductive_val --negative_sampling_ratio_train 3 --n_epochs 500

train-coauthor-cs:
	python train_inductive.py  --dataset coauthor-cs --transductive_val --negative_sampling_ratio_train 2 --n_epochs 200

train-pubmed:
	python train_inductive.py  --dataset pubmed --transductive_val --negative_sampling_ratio_train 2 --n_epochs 200

train-computers:
	python train_inductive.py  --dataset amazon-computers --transductive_val --negative_sampling_ratio_train 2 --n_epochs 100

train-photos:
	python train_inductive.py  --dataset amazon-photos --transductive_val --negative_sampling_ratio_train 1.2 --n_epochs 100

train-burglary:
	python train_inductive.py  --dataset inp-burglary --transductive_val --negative_sampling_ratio_train 1.2 --n_epochs 500

train-burglary-lambdas-thetas:
	python train_inductive.py  --dataset inp-burglary --transductive_val --negative_sampling_ratio_train 1.2 --n_epochs 500

train-burglary-ind-val:
	python train_inductive.py  --dataset inp-burglary --inductive_val --negative_sampling_ratio_train 1.2 --n_epochs 500

train-burglary-ind-val-lambdas-thetas:
	python train_inductive.py  --dataset inp-burglary --inductive_val --negative_sampling_ratio_train 1.2 --n_epochs 500

train-burglary-no-augmented-data:
	python train_inductive_no_augmented_data.py  --dataset inp-burglary --transductive_val --n_epochs 500