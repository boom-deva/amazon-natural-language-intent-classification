# Data Description

### Download URL

Dataset is available for download from https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US.

### Data Overview

Dataset is in JSON format. Each sample cotains an utterance in text, a label in text, and a label in numbers. A few examples are shown below:

| label | text                             | label_text        |
| ----- | -------------------------------- | ----------------- |
| 48    | wake me up at nine am on friday  | alarm_set         |
| 46    | olly pause for ten seconds       | audio_volume_mute |

There are 11,514 training samples, 2,033 development samples, and 2,974 testing samples.