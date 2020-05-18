.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = tennis
PYTHON_INTERPRETER = python3
USE_GPU = 0
ifeq ($(USE_GPU), 1)
	CUDA_DEVICE = 0
else
	CUDA_DEVICE = -1
endif
DATA_DIR = $(PROJECT_DIR)/data

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

## Run tests
RUNTEST=python -m unittest -v -b

ALLMODULES=$(wildcard src/test/test_*.py)

test_all:
	${RUNTEST} ${ALLMODULES}

## Make Dataset
data: requirements
	mkdir -p $(DATA_DIR)/processed
	mkdir -p $(DATA_DIR)/interim
	mkdir -p $(DATA_DIR)/raw

ALL_VIDEOS=$(wildcard $(DATA_DIR)/raw/*.mp4)

score_extract: $(addprefix $(DATA_DIR)/interim/scoreboard/, $(addsuffix .pkl, $(basename $(notdir $(ALL_VIDEOS)))))
$(DATA_DIR)/interim/scoreboard/%.pkl: $(DATA_DIR)/processed/frames/%
	python deeptennis/features/extract_scoreboard.py \
	--save-path $@ \
	--frames-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) \
	--meta-file $(PROJECT_DIR)/deeptennis/match_meta.json

score_mask_extract: $(addprefix $(DATA_DIR)/interim/score_mask/, $(basename $(notdir $(ALL_VIDEOS))))
$(DATA_DIR)/interim/score_mask/%: $(DATA_DIR)/processed/frames/%
	python deeptennis/features/extract_scoreboard.py \
	--save-path $@ \
	--frames-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) \
	--meta-file $(PROJECT_DIR)/deeptennis/match_meta.json \
	--segmentation 1

court_extract: $(addprefix $(DATA_DIR)/interim/court/, $(addsuffix .json, $(basename $(notdir $(ALL_VIDEOS)))))
$(DATA_DIR)/interim/court/%.json: $(DATA_DIR)/interim/action_mask/%.json
	python deeptennis/features/extract_court_keypoints.py \
	--mask-path $< \
	--save-path $@ \
	--frames-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) \
	--meta-file $(PROJECT_DIR)/metadata/match_meta.json

player_tracking: $(addprefix $(DATA_DIR)/interim/player_tracking/, $(addsuffix .json, $(basename $(notdir $(ALL_VIDEOS)))))
$(DATA_DIR)/interim/player_tracking/%.json: $(DATA_DIR)/processed/frames/%
	mkdir -p $(DATA_DIR)/interim/player_tracking && \
	$(PYTHON_INTERPRETER) scripts/make_json_dataset.py --frames-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) --output-path /tmp/player_tracking_temp.json && \
	allennlp predict $(MODEL_PATH) /tmp/player_tracking_temp.json \
	--cuda-device $(CUDA_DEVICE) --output-file $@ --silent --predictor default_image \
	--batch-size 4 \
	--overrides '{"dataset_reader": {"type": "image_annotation", "augmentation": [{"type": "resize","height": 512, "width": 512}, {"type": "normalize"}], "lazy": true}, "model": {"roi_box_head": {"decoder_thresh": 0.01}}}' \
	--include-package allencv.data.dataset_readers \
	--include-package allencv.modules.image_encoders \
	--include-package allencv.modules.image_decoders \
	--include-package allencv.modules.im2vec_encoders \
	--include-package allencv.modules.im2im_encoders \
	--include-package allencv.models.object_detection \
	--include-package allencv.predictors \
	&& rm /tmp/player_tracking_temp.json

data/interim/tracking_videos/%: $(DATA_DIR)/interim/player_tracking/%.json
	python $(PROJECT_DIR)/scripts/make_tracking_video.py \
	--tracking-path $(addsuffix .json, $(DATA_DIR)/interim/player_tracking/$(basename $(notdir $<))) \
	--frame-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) \
	--save-path $@

clip_videos: $(addprefix $(DATA_DIR)/interim/match_clips_video/, $(notdir $(ALL_VIDEOS)))
$(DATA_DIR)/interim/match_clips_video/%.mp4: $(DATA_DIR)/interim/clips/%.pkl
	python deeptennis/data/clips2vid.py \
	--clip-path $< \
	--save-path $@ \
	--frame-path $(DATA_DIR)/processed/frames/$(basename $(notdir $<)) \

.PRECIOUS: $(DATA_DIR)/interim/action_mask/%.json
$(DATA_DIR)/interim/action_mask/%.json: $(DATA_DIR)/interim/featurized_frames/%.npy
	python $(PROJECT_DIR)/scripts/extract_action.py \
	--features-path $< \
	--save-path $@

featurized: $(addprefix $(DATA_DIR)/interim/featurized_frames/, $(addsuffix .npy, $(basename $(notdir $(ALL_VIDEOS)))))
.PRECIOUS: $(DATA_DIR)/interim/featurized_frames/%.npy
$(DATA_DIR)/interim/featurized_frames/%.npy : FEATURIZE_PCA = 10
$(DATA_DIR)/interim/featurized_frames/%.npy : BATCH_SIZE = 32
$(DATA_DIR)/interim/featurized_frames/%.npy : $(DATA_DIR)/processed/frames/%
	python $(PROJECT_DIR)/scripts/featurize_frames.py \
	--img-path $< \
	--save-path $@ \
	--gpu $(USE_GPU) \
	--batch-size $(BATCH_SIZE) \
	--pca $(FEATURIZE_PCA)

frames: VFRAMES = 2000
frames: FPS = 1
frames: $(addprefix $(DATA_DIR)/processed/frames/, $(basename $(notdir $(ALL_VIDEOS))))
.PRECIOUS: $(DATA_DIR)/processed/frames/%
$(DATA_DIR)/processed/frames/%: $(DATA_DIR)/raw/%.mp4
	python $(PROJECT_DIR)/scripts/vid2img.py \
	--vid-path $< \
	--img-path $@ \
	--fps $(FPS) \
	--vframes $(VFRAMES)

clean_data_interim:
	rm -rf $(DATA_DIR)/interim/*

clean_data_processed:
	rm -rf $(DATA_DIR)/processed/*

clean_data: clean_data_interim clean_data_processed


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 deeptennis

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
