# Third-party model licenses

The MIT license in [LICENSE](LICENSE) covers the source code in this repository
and nothing else.

It does not cover the machine-learning models this tool uses, and it cannot: no
model weights are stored in or distributed by this repository. Each model is
downloaded from Hugging Face at runtime under its own terms.

| Model | License | Commercial use |
| --- | --- | --- |
| [`pangram/editlens_Llama-3.2-3B`](https://huggingface.co/pangram/editlens_Llama-3.2-3B) | CC BY-NC-SA 4.0 | **Prohibited** |
| [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B) | Llama 3.2 Community License | Conditional — see the license |
| [`ShantanuT01/vanguard-ai-text-detector`](https://huggingface.co/ShantanuT01/vanguard-ai-text-detector) | MIT | Permitted |
| [`ogmatrixllm/glyph-v1.1`](https://huggingface.co/ogmatrixllm/glyph-v1.1) | See its model card | See its model card |

EditLens forbids commercial use outright, so **running this tool as configured
is a personal, non-commercial activity** regardless of the permissive license on
the code. A permissive grant on a wrapper cannot widen the terms of what it
wraps.

If you need a commercially usable build, you would have to drop EditLens and
re-check the terms of whatever replaces it.

Model licenses change. Check each model card yourself before relying on any of
this; the table above records what those cards said when it was written and is
not legal advice.
