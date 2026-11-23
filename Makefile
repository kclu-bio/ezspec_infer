NVIDIA_API_KEY='YOUR_API_KEY'
BASE_DIR=/home/kechenlu/kechenlu_link/RAPLM2/baseline_model/ezspec
DATA_DIR=$(BASE_DIR)/data_dir

VN_BASE=$(DATA_DIR)/screen/vergne_nitrilase
VN_PRED=$(VN_BASE)/boltz_outputs/boltz_results_boltz_inputs/predictions
VN_PDB=$(VN_BASE)/boltz_outputs/boltz_results_boltz_inputs/pdb_files

CN_BASE=$(DATA_DIR)/screen/catnip
CN_PRED=$(CN_BASE)/boltz_outputs/boltz_results_boltz_inputs/predictions
CN_PDB=$(CN_BASE)/boltz_outputs/boltz_results_boltz_inputs/pdb_files

ALL_SPLIT_BASE=$(DATA_DIR)/esibank/all_split
ALL_SPLIT_CSV=$(ALL_SPLIT_BASE)/all_split.csv
ALL_SPLIT_PRED=$(ALL_SPLIT_BASE)/boltz_outputs/boltz_results_boltz_inputs/predictions
ALL_SPLIT_PDB=$(ALL_SPLIT_BASE)/boltz_outputs/boltz_results_boltz_inputs/pdb_files
	
preprocess:
	python $(BASE_DIR)/src/make_screening_inputs.py \
	  -training-dir $(DATA_DIR)/esibank \
	  -screening-dir $(DATA_DIR)/screen \
	  -api-key "$(NVIDIA_API_KEY)"\

VN_boltz:
	boltz predict \
	$(VN_BASE)/boltz_inputs \
	--out_dir $(VN_BASE)/boltz_outputs \
	--cache /home/kechenlu/kechenlu_link/.cache/boltz

VN_cif_convert:
	python $(BASE_DIR)/src/convert.py --input-dir $(VN_PRED) --output-dir $(VN_PDB)

VN_gather:
	python $(BASE_DIR)/src/gather_inputs_screening.py -pdb-dir $(VN_PDB) -output-dir $(VN_BASE) \
	-data-df $(VN_BASE)/data.csv -substrate-file $(VN_BASE)/Substrates.csv

VN_process:
	python $(BASE_DIR)/src/preprocess.py \
	--prepared_data_dir $(VN_BASE) \
	--features_output_dir $(VN_BASE) \
	--src_root_dir $(BASE_DIR) \
	--data_root_dir $(DATA_DIR)/ezspec_data

VN_inference:
	python $(BASE_DIR)/src/run_inference.py \
	--src-root-dir $(BASE_DIR) \
	--logging-dir $(BASE_DIR)/data_dir/eval/outputs/logs \
	--features-output-dir $(VN_BASE) \
	--prepared-data-dir $(VN_BASE) \
	--ckpt-path $(BASE_DIR)/model_ckpts/best-checkpoint.ckpt

catnip_boltz:
	boltz predict \
	$(CN_BASE)/boltz_inputs \
	--out_dir $(CN_BASE)/boltz_outputs \
	--cache /home/kechenlu/kechenlu_link/.cache/boltz

catnip_cif_convert:
	python $(BASE_DIR)/src/convert.py --input-dir $(CN_PRED) --output-dir $(CN_PDB)

catnip_gather:
	python $(BASE_DIR)/src/gather_inputs_screening.py -pdb-dir $(CN_PDB) -output-dir $(CN_BASE) \
	-data-df $(CN_BASE)/data.csv -substrate-file $(CN_BASE)/Substrates.csv

catnip_process:
	python $(BASE_DIR)/src/preprocess.py \
	--prepared_data_dir $(CN_BASE) \
	--features_output_dir $(CN_BASE) \
	--src_root_dir $(BASE_DIR) \
	--data_root_dir $(DATA_DIR)/ezspec_data

catnip_inference:
	python $(BASE_DIR)/src/run_inference.py \
	--src-root-dir $(BASE_DIR) \
	--logging-dir $(BASE_DIR)/data_dir/eval/outputs/logs \
	--features-output-dir $(CN_BASE) \
	--prepared-data-dir $(CN_BASE) \
	--ckpt-path $(BASE_DIR)/model_ckpts/best-checkpoint.ckpt

ALL_SPLIT_input:
	python $(BASE_DIR)/src/make_test_input.py \
	--input-csv $(ALL_SPLIT_CSV) \
	--output-dir $(ALL_SPLIT_BASE) \
	--api-key $(NVIDIA_API_KEY)

ALL_SPLIT_boltz:
	boltz predict \
	$(ALL_SPLIT_BASE)/boltz_inputs \
	--out_dir $(ALL_SPLIT_BASE)/boltz_outputs \
	--cache /home/kechenlu/kechenlu_link/.cache/boltz

ALL_SPLIT_cif_convert:
	python $(BASE_DIR)/src/convert.py --input-dir $(ALL_SPLIT_PRED) --output-dir $(ALL_SPLIT_PDB)

ALL_SPLIT_gather:
	python $(BASE_DIR)/src/gather_inputs_screening.py -pdb-dir $(ALL_SPLIT_PDB) -output-dir $(ALL_SPLIT_BASE) \
	-data-df $(ALL_SPLIT_BASE)/data.csv -substrate-file $(ALL_SPLIT_BASE)/Substrates.csv

ALL_SPLIT_process:
	python $(BASE_DIR)/src/preprocess.py \
	--prepared_data_dir $(ALL_SPLIT_BASE) \
	--features_output_dir $(ALL_SPLIT_BASE) \
	--src_root_dir $(BASE_DIR) \
	--data_root_dir $(DATA_DIR)/ezspec_data

ALL_SPLIT_inference:
	python $(BASE_DIR)/src/run_inference.py \
	--src-root-dir $(BASE_DIR) \
	--logging-dir $(BASE_DIR)/data_dir/eval/outputs/logs \
	--features-output-dir $(ALL_SPLIT_BASE) \
	--prepared-data-dir $(ALL_SPLIT_BASE) \
	--ckpt-path $(BASE_DIR)/model_ckpts/best-checkpoint.ckpt


