from transformers import Wav2Vec2Processor, HubertModel
from matplotlib import pyplot as plt
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

# print(ds["speech"][0])

plt.plot(ds["speech"][0])
plt.show()
input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
print(input_values.shape)

model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
print(model.config)
hidden_states = model(input_values).last_hidden_state
print(hidden_states.shape)
