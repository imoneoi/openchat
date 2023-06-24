# Make it more memory efficient by monkey patching the model with FlashAttn
# Need to call this before importing transformers.
from ochat.patches.llama_attn_monkey_patch import replace_llama_attn
from ochat.patches.starcoder_attn_monkey_patch import replace_starcoder_attn

replace_llama_attn()
replace_starcoder_attn()

print ("OpenChat patches applied.")
