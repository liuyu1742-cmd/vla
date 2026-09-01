# OpenVLA Local Inference Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a PyCharm-visible, isolated CUDA environment able to run a local OpenVLA action prediction.

**Architecture:** A dedicated Conda environment owns OpenVLA's Python/CUDA dependency set. Existing RoboCasa environments remain unchanged. A read-only verifier checks CUDA and model presence before a smoke command loads the local model.

**Tech Stack:** Conda; Python 3.10; PyTorch 2.2.2 CUDA 12.1; Transformers 4.40.1; vendored OpenVLA; PyCharm.

## Global Constraints

- Do not access or alter `datasets/oxe/bridge_orig`.
- Do not alter the `robocasa` or `robosim` Conda environments.
- Pin `torch==2.2.2`, `torchvision==0.17.2`, `transformers==4.40.1`, `tokenizers==0.19.1`, and `timm==0.9.10`.
- Do not install FlashAttention for this Windows baseline.
- Do not claim inference success without an actual local `predict_action` result.

---

### Task 1: Create the isolated OpenVLA CUDA environment

**Files:**

- Create: `requirements-openvla-inference-win.txt`
- Create: `docs/OPENVLA_PYCHARM_SETUP.md`

**Interfaces:**

- Produces Conda environment `openvla` at `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla`.

- [ ] **Step 1: Create `requirements-openvla-inference-win.txt`**

```text
accelerate>=0.25.0
einops
huggingface_hub
peft==0.11.1
protobuf
sentencepiece==0.1.99
timm==0.9.10
tokenizers==0.19.1
transformers==4.40.1
```

- [ ] **Step 2: Create and populate the environment**

Run `conda create -n openvla python=3.10 -y`.

Run `conda run -n openvla python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121`.

Run `conda run -n openvla python -m pip install -r requirements-openvla-inference-win.txt`.

- [ ] **Step 3: Install the vendored source**

Run `conda run -n openvla python -m pip install -e third_party/openvla --no-deps`.

Run `conda run -n openvla python -c "import prismatic; print(prismatic.__file__)"`.

Expected: command exits 0 and prints a path under `third_party/openvla`.

### Task 2: Add a deterministic environment verifier

**Files:**

- Create: `tools/check_openvla_environment.py`
- Create: `tests/test_check_openvla_environment.py`

**Interfaces:**

- Produces `build_report(model_dir: pathlib.Path) -> dict[str, object]`.

- [ ] **Step 1: Write the failing test**

```python
def test_report_marks_missing_checkpoint(tmp_path):
    report = build_report(tmp_path)
    assert report["model_files_present"] is False
    assert "cuda_available" in report
```

- [ ] **Step 2: Run the test and observe failure**

Run `conda run -n openvla python -m pytest tests/test_check_openvla_environment.py -q`.

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the report**

`build_report` records Python/PyTorch version, CUDA availability, GPU name, whether `model_dir/config.json` exists, and whether `torch.ones((16,16), device="cuda") @ torch.ones((16,16), device="cuda")` is finite.

- [ ] **Step 4: Verify the test and actual environment**

Run `conda run -n openvla python -m pytest tests/test_check_openvla_environment.py -q`.

Run `conda run -n openvla python tools/check_openvla_environment.py`.

Expected: tests pass; JSON reports CUDA availability and the RTX 3090.

### Task 3: Expose the verifier in PyCharm

**Files:**

- Create: `.run/OpenVLA CUDA Environment Check.run.xml`
- Modify: `docs/OPENVLA_PYCHARM_SETUP.md`

**Interfaces:**

- Consumes PyCharm interpreter `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe`.
- Produces Run-menu configuration `OpenVLA CUDA Environment Check` that executes `tools/check_openvla_environment.py` at `$PROJECT_DIR$`.

- [ ] **Step 1: Add the shared run configuration**

```xml
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="OpenVLA CUDA Environment Check" type="PythonConfigurationType" factoryName="Python">
    <module name="OpenVLA-Simulator" />
    <option name="SCRIPT_NAME" value="$PROJECT_DIR$/tools/check_openvla_environment.py" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="SDK_HOME" value="C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe" />
    <method v="2" />
  </configuration>
</component>
```

- [ ] **Step 2: Document interpreter selection**

Document: open `File > Settings > Project: OpenVLA-Simulator > Python Interpreter`, add and select the existing Conda interpreter, then choose the configuration from PyCharm's Run menu.

- [ ] **Step 3: Verify the same interpreter directly**

Run `C:\\Users\\sjtu101\\miniconda3\\envs\\openvla\\python.exe tools/check_openvla_environment.py`.

Expected: exit 0 with the same JSON as Task 2.

### Task 4: Run one local OpenVLA action-prediction smoke test

**Files:**

- Create: `tools/run_openvla_inference_smoke.py`
- Create: `tests/test_run_openvla_inference_smoke.py`
- Create: `outputs/openvla_inference_smoke_report.json`

**Interfaces:**

- Consumes local model directory and instruction string.
- Produces JSON with model path, instruction, device, dtype, and a seven-element `predicted_action` list.

- [ ] **Step 1: Write a failing validation test**

```python
def test_requires_local_model_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        validate_model_dir(tmp_path)
```

- [ ] **Step 2: Run test to observe failure**

Run `conda run -n openvla python -m pytest tests/test_run_openvla_inference_smoke.py -q`.

- [ ] **Step 3: Implement the smoke command**

Require `config.json`; load `AutoProcessor` and `AutoModelForVision2Seq` locally with `trust_remote_code=True`; generate a local RGB placeholder; call `predict_action(..., unnorm_key="bridge_orig", do_sample=False)` using `In: What action should the robot take to pick up the cup?\nOut:`; write JSON. The script must not access the BridgeData directory.

- [ ] **Step 4: Execute and inspect the report**

Run `conda run -n openvla python -m pytest tests/test_run_openvla_inference_smoke.py -q`.

Run `conda run -n openvla python tools/run_openvla_inference_smoke.py --model-dir models/openvla-7b --report outputs/openvla_inference_smoke_report.json`.

Expected: tests pass and output has a seven-element action.
