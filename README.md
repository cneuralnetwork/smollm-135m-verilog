# SmolLM-135M in SystemVerilog

A prompt-to-text inference engine for [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M). The model path—from byte-level BPE tokenization through 30 decoder layers, KV cache, greedy token selection, and UTF-8 output—is implemented in **SystemVerilog**. A local Prompt Studio dashboard starts the Verilog executable and displays progress and output.

**This is a simulator implementation, not synthesizable FPGA or ASIC RTL.** Verilator compiles the SystemVerilog source into a local executable. The dashboard's small Node server provides HTTP transport; it does not run the model or tokenize text.

## What is included

| Path | Purpose |
| --- | --- |
| `smollm.sv` | SystemVerilog tokenizer, full transformer inference, cache, and decoder |
| `data/*.mem` | Prepared vocabulary, BPE merge, Unicode class, and tensor offset tables |
| `data/tokenizer.json`, `data/config.json` | Original model metadata |
| `dashboard/server.mjs` | Local-only process bridge and HTTP server |
| `dashboard/public/` | Prompt Studio UI and bundled Inter / Space Grotesk fonts |
| `data/model.safetensors` | **Downloaded separately**; excluded from Git |

The checkpoint has 30 layers, hidden size 576, 9 query heads, 3 key/value heads, 1,536 feed-forward channels, a tied 49,152-token embedding, and a 2,048-token context limit. Inference uses F32 checkpoint weights and Verilog `real` arithmetic.

## Requirements

- **Verilator 5.054 or newer**, a C++ compiler, and `make`.
- **Node.js 20 or newer** for the dashboard. Node is not needed for the command-line Verilog run.
- About **1 GB of free disk space** for the 538 MB checkpoint and generated build files.
- Enough RAM for the simulator (about **410 MB allocated** in the tested one-token run, plus process overhead).
- macOS or Linux. Commands below use a POSIX shell.

On macOS with Homebrew:

```sh
brew install verilator node
```

Check the tools:

```sh
verilator --version
node --version
```

## 1. Clone and download the checkpoint

```sh
git clone https://github.com/cneuralnetwork/smollm-135m-verilog.git
cd smollm-135m-verilog
mkdir -p data
curl -fL --retry 3 \
  'https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors' \
  -o data/model.safetensors
```

The expected file size is **538,090,408 bytes**. Check its SHA-256 hash:

```sh
shasum -a 256 data/model.safetensors
```

Expected:

```text
c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093  data/model.safetensors
```

The static `tensor_offset.mem` table targets this exact checkpoint. If the hash differs, download the checkpoint revision used here before running inference.

## 2. Run from the terminal

Compile once, from the repository root:

```sh
verilator --binary --timing -O3 -Wno-fatal -j 4 smollm.sv
```

Generate one token:

```sh
./obj_dir/Vsmollm '+prompt=Hi' +max_new=1
```

The tested result is `OUTPUT: ,`. For a longer continuation:

```sh
./obj_dir/Vsmollm '+prompt=The capital of France is' +max_new=8
```

Quote the entire `+prompt=...` argument when it contains spaces or shell punctuation. `+max_new` defaults to 16. `+bos` explicitly prepends the model's BOS token; ordinary nonempty prompts do not get one. `+tokenize_only` prints token IDs quickly without loading the checkpoint:

```sh
./obj_dir/Vsmollm '+prompt=Once upon a time,' +tokenize_only
```

The simulator prints progress for each prompt token. Generation stops at EOS or the requested token count. Prompt plus requested output must fit within 2,048 tokens.

## 3. Run the dashboard

From the repository root:

```sh
node dashboard/server.mjs
```

Open **<http://127.0.0.1:4173>** in your browser. The server compiles the Verilog source automatically if the executable is missing or stale. It listens only on `127.0.0.1` and starts one generation at a time.

- **Workspace** accepts a prompt, shows per-token progress, streams generated text, and has Stop and Copy controls.
- **History** keeps completed prompts and outputs in this browser's local storage.
- **Settings** controls the maximum generated tokens (1–64) and optional BOS insertion.

To choose another port:

```sh
PORT=4180 node dashboard/server.mjs
```

Stop the server with **Ctrl+C**. The dashboard cannot be hosted as a static page and still reach a Verilog process on your computer; the local Node bridge is required.

## How it works

```text
prompt → Verilog Unicode/ByteLevel split → BPE merges → token IDs
       → tied embedding → 30 Llama decoder layers with RoPE and GQA
       → final RMSNorm → tied vocabulary projection → greedy token ID
       → Verilog byte decoder → terminal text / dashboard stream
```

Each decoder layer applies RMSNorm, Q/K/V projections, rotary position embeddings, grouped-query attention with a KV cache, an output projection, then another RMSNorm and a SiLU-gated MLP. The checkpoint's `model.safetensors` tensor offsets are stored in `data/tensor_offset.mem`. The tied embedding is kept in memory, while other weight matrices stream by layer.

The dashboard asks the executable for `+machine` output. That emits token bytes as hex records; the browser displays the UTF-8 text. Node does not calculate logits or choose tokens.

## Limits and troubleshooting

- **Slow output:** On the tested Mac, one prompt token takes roughly **20 seconds**. Six prompt tokens can take around **two minutes before the first output token**. This is expected for 135M parameters in a simulator. The dashboard shows the current token.
- **`cannot open data/model.safetensors`:** Download the checkpoint and run commands from the repository root.
- **Hash mismatch:** The static tensor offsets may point at the wrong bytes. Use the checkpoint hash listed above.
- **Port already in use:** Run `PORT=4180 node dashboard/server.mjs` and open the new URL.
- **No Verilator:** Install it and confirm `verilator --version` before starting the dashboard.
- **Prompt too long:** Shorten it or reduce `+max_new`; the model context is 2,048 tokens.
- **Output quality:** This is the **base** SmolLM checkpoint, so it continues text rather than following chat instructions. Decoding is greedy; sampling, temperature, and batching are not implemented.
- **Hardware:** `real`, strings, and simulation file I/O are used. This source cannot be synthesized into an FPGA/ASIC as written.

## Verification

This project was checked with Verilator 5.054. The tokenizer matched the checkpoint tokenizer on prompts covering spaces, contractions, digits, UTF-8 text, and special tokens. Independent Torch calculations agreed with Verilog next-token results: `Hi` → `,` and `The capital of France is` → ` Paris`. The dashboard's HTTP stream and browser UI were tested end to end with `Hi` → `,`.

## License and credits

Project code is Apache-2.0; see [LICENSE](LICENSE). The original [SmolLM-135M model and tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM-135M) are Apache-2.0. Bundled [Inter](https://github.com/google/fonts/tree/main/ofl/inter) and [Space Grotesk](https://github.com/google/fonts/tree/main/ofl/spacegrotesk) are under the SIL Open Font License; their license texts are in `dashboard/public/fonts/`. The dashboard layout follows the [Prompt Studio reference artifact](https://claude.ai/artifact/KbzxjjB53ekn32juENXaS3#play-15dd8cd198dd).
