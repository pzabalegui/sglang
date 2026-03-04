from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import os
os.environ['TRUST_REMOTE_CODE'] = 'true'

print('=== Conversión a FP8 con llm-compressor ===')

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"]
)

oneshot(
    model="/root/glm47_abliterated_final",
    output_dir="/root/glm47_abliterated_fp8",
    recipe=recipe,
)

print('Conversión completada!')
