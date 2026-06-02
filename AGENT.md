# Agent Instructions

## [IMPORTANT] Correct symlinked paths for display
When giving me absolute paths, e.g. when pointing to result files, always give me paths that start in `/home-local/${user}.nobkp/` rather than `/gel/usr/${user}/` please.

## [IMPORTANT] Make all changes in separate worktrees
You should not make this change in this worktree ever. 
Instead, create a separate worktree, placed into `worktrees/`, with an appropriate name based on the feature you are implementing.
Run the README install instructions to re-create a local environment, and always work in that environment; make sure to pull submodules and build your own version of the raytracer.
However, NEVER re-run the download for the datasets. Instead, symlink `data/` to the main clone's `data/` e.g. `/gel/usr/${user}/Desktop/gray/data`. 
If you need to modify the data in anyway, or re-run colmap, image resizing, or initial dense point cloud creation, please instead create a local copy of `data/`.
If you just need one or more pretrained scenes, copy the ones alredy downloaded in `output/pretrained`.
Only copy as many scenes as you need to avoid excess memory use.

## [IMPORTANT] Use the `uv` package manager at all times
This project uses the `uv` package manager. 
New packages should be tracked with `uv add`.
Temporary installs should use `uv pip install`. 
DON'T CALL `pip` DIRECTLY. 

Good example:
```
uv add numpy # Correct! using uv to track new package
```

Bad example:
```
pip install numpy # X WRONG, don't call pip directly
```

Bad example:
```
pip uninstall numpy # X WRONG, don't call pip directly even for uninstalling 
```

Bad example:
```
python -m pip ... # X WRONG, always go through uv for any pip command
```

If you need to do a temporary install that you don't want to track, use `uv pip install` but be careful not to use `pip` directly.

Good example:
```
uv pip install seaborn # Correct! using uv pip for temporary install that does not need to be tracked 
```

But please always use `uv add` when this dependency is needed for non-throwaway code. 

## [IMPORTANT] Before doing anything, first activate the environment 
Immediately run `source .venv/bin/activate` before anything else.
ITS CRITICAL THAT YOU DO SO BEFORE RUNNING ANY COMMAND, EVEN IF IT SEEMS POINTLESS.

Bad example:
```
cd src/ # X WRONG, env wasn't activated
```

Bad example:
```
source .venv/bin/activate` && cd src/ # X WRONG, don't activate on the same line
```

Good example:
```
source .venv/bin/activate` # Correct! activate first, and only once
cd src/
```

The environment is provided for you: you do not need to recreate it or install dependencies, just activate it.

Bad example:
```
uv sync # X WRONG, the environment is provided, don't need to recreate it.
```

This environment is shared with other agents, please do not modify it by installing or uninstalling packages. 

Bad example:
```
pip uninstall torch # X WRONG, the environment is provided, don't touch existing dependencies.
pip install --upgrade torch # X WRONG, don't touch existing dependencies
```

## [IMPORTANT] CUDA code compilation
When you modify the CUDA code, compile with `bash make.sh` to verify its OK. 
You only need to compile once after making changes.

Good example:
```
# <edit the code>
bash make.sh # Correct! compile CUDA code after changes
bash run.sh ...
bash run.sh ... # No need to compile again if no further CUDA changes
```

Bad example:
```
# <edit the code>
bash run.sh ... # X WRONG, running without compiling after CUDA changes
```

Bad example:
```
# <edit the code>
bash make.sh # Correct! compile CUDA code after changes
bash make.sh && bash run.sh ... # X WRONG, no need to compile again if no further CUDA changes
```

Bad example:
```
# <edit the Python code only>
bash make.sh # X WRONG, no need to compile if only Python code changed
```

You do not need to compile initially, as a clean build is already provided.

Bad example:
```
# <no code changes yet>
bash make.sh # X WRONG, no need to compile at the start
```

## [IMPORTANT] Use the provided `run.sh` utility for running experiments
Please use the run.sh utility for running experiments instead of calling `train.py` directly.
The `run.sh` utility provides logging and evaluation; I want these for all your runs, even if you don't need them directly yourself.

Good example:
```
bash run.sh tmp/bicycle_test -s data/360_v2/bicycle -c lq -r 8 # Correct! using the run.sh utility
```

Bad example:
```
python train.py -s data/360_v2/bicycle -m tmp/bicycle_test -c lq -r 8 # X WRONG, not using the run.sh utility
```

## [IMPORTANT] Use the GPU responsibly and queue tasks
You are sharing the GPU with me and other tasks. 
This means we need to be very careful about how we use it to avoid conflicts, OOM errors, and to ensure efficient usage.

### Always use the `pueue` queue for GPU tasks
For any task that requires GPU, you MUST use the `pueue` queue.
The general idea is to prepend your command with `pueue add --print-task-id -- '...'` to add it to the queue and get the task id. 

IT IS CRITICAL THAT YOU ALWAYS USE THE QUEUE, EVEN IF IT SEEMS ANNOYING OR POINTLESS.

Bad example:
```
python train.py -s data/360_v2/bicycle -c lq -r 8 # X WRONG, not using the queue
```

Good example:
```
pueue add --print-task-id -- 'python train.py -s data/360_v2/bicycle -c lq -r 8' # Correct! using the queue
```

Bad example:
```
bash run.sh tmp/bicycle_test -s data/360_v2/bicycle -c lq -r 8 # X WRONG, not using the queue for a bash script that uses the GPU
```

Good example:
```
pueue add --print-task-id -- 'bash run.sh tmp/bicycle_test -s data/360_v2/bicycle -c lq -r 8' # Correct! using the queue
```

Use `pueue wait` to wait for all your tasks to complete. 
This will automatically wait for your your task to both start, and finish. 
Using `pueue wait` is the most efficient way of getting notified when your task is done and you can continue working.

Bad example:
```
pueue add --print-task-id -- 'bash run.sh tmp/bicycle_test -s data/360_v2/bicycle -c lq -r 8' 
# => New task added (id <task-id>).
sleep 6000
stat tmp/bicycle_test/psnr.csv # X WRONG, inefficiently guessing how long it will take and then checking if the task is done.
```

Good example:
```
pueue add --print-task-id -- 'bash run.sh tmp/bicycle_test -s data/360_v2/bicycle -c lq -r 8'
# => New task added (id <task-id>).
pueue wait <task_id> # Correct! waiting for the task to complete efficiently.
cat tmp/bicycle_test/psnr.csv # Now you can check the results after the task is done.
```

If you need to kill a task, use `pueue kill <task_id>` but only for your own tasks.

Good example:
```
pueue add --print-task-id -- '...'
# => New task added (id <task-id>).
pueue kill <task_id> # Correct! killing your own task if needed
```

Bad example:
```
pueue add --print-task-id -- '...'
# => New task added (id <task-id>).
pueue kill <other_task_id> # X WRONG, never kill other agents' tasks
```

Waiting my take a while. That is OK -- BE PATIENT.

Finally, always use quotation marks around the command you pass to `pueue add` to avoid issues with special characters.

Good example:
```
pueue add --print-task-id -- 'mkdir tmp2 && cd tmp2 && python ...' # Correct! using quotation marks
```

Bad example:
```
pueue add --print-task-id -- mkdir tmp2 && cd tmp2 && python ...# X WRONG, will fail command parsing
```

### Do not kill or interfere with other agents or my tasks
You CANNOT kill tasks using `pueue kill` or `kill` if they are not yours, and you CANNOT block the queue by running GPU tasks outside of `pueue`.
You CANNOT stop the queue using `pueue shutdown`.
NEVER break these rules as they are essential; doing so is as bad a complete failure of your task.

### Maximize GPU efficiency
Train in low quality mode (`-c lq`) to get faster feedback, and reduce resolution with `-c 8`. For final runs, use `-r 4`. No need to even run `hq`, normal quality is good enough for final runs. SAVE IN `./tmp/<TASK_NAME>/<EXPERIMENT_NAME>` AND PICK A DESCRIPTIVE NAME FOR THE RUN. Use the same task name for all runs related to the same task.
  * Example debug command: `pueue add --print-task-id -- 'bash run.sh tmp/<TASK_NAME>/<EXPERIMENT_NAME> -y -s data/360_v2/bicycle -c lq -r 8'`. 
  * Example final command: `pueue add --print-task-id -- 'bash run.sh tmp/<TASK_NAME>/<EXPERIMENT_NAME> -y -s data/360_v2/bicycle -r 4'`.
You only need to do final runs when PSNR is required for your task.
- NEVER RUN FULL RESOLUTION, always `-r 4` at most.
- Use the yes (`-y`) if you need to overwrite existing runs.
- If there is no baseline, run one in `tmp/baselines/bicycle`. You should check if it exists first.

Please always use the `run.sh` over calling `train.py` directly, as it gives eval metrics we might want to inspect later, and is not wasteful as the rendering and evaluation runs really fast. Do not hesitate to add proper logging even if it slows things down minimally.

Correctness remains essential: do use the GPU for validation. 
Do not skimp on running GPU tasks that you required for validation under the guise of "saving GPU time".
You must use GPU tasks to validate the success of your changes, even if this requires several runs. That's OK.

## [IMPORTANT] Final reports
Write a summary of your changes in `ai_reports/$TASK_NAME.md`. Before modifying an existing feature, consult existing AI reports. Only produce reports for big tasks (new features, code changes), not for small verifications or questions about the code. Reports should:
  * Include a concise description of your changes
  * A clear "WARNINGs" section if you believe something could be incorrect or if you had to disrespect the prompt out of absolute necessity, or if you had to change default settings or any behavior in other code paths. 
  * Include equations and all details of your implementation, but try to keep it short (explain *what* you did in all its details, but not *how*). 
  * If you had to make a design decision, include the alternatives you considered and why you chose the one you did.
  * If your task included running experiments, include final PSNR in a "Results" section alongside final rendered test view 0 images, as compared to the baseline. Put images in `ai_reports/${TASK_NAME}_images/`. If you had to run ablations, include a table of results for the ablations as well. 

## [IMPORTANT] Additional project instructions
- Use the scenes that are already provided in the `data/` directory. 
- When explicitly asked to consult papers or codebases, download them to `ai_references` (git clone or wget the pdf from arxiv), and reach from there. Make sure they weren't already fetched before doing so. Don't just read them online and summarize, actually read them in depth locally and understand them. 
- When adding logging images, try to minimize spam by logging them only at the `--preview_iterations`.

## General Instructions
- Dumb code is better: 
    * Write straightlight code with top to down control flow as much as possible.
    * Large functions are perfectly fine.
    * Avoid extraction to helper function unless really required.
    * Don't create main() functions, write directly after `if __name__ == "__main__"`.
    * Feel free to repeat yourself locally in the same file. Repetition across faraway files is a problem, but local repetition is not.
    
- Minimal AI comments, respect human comments:
    * ADD AS FEW COMMENTS AS POSSIBLE, THE COMMENTS SHOULD "LOOK" NORMAL AND FIT INTO THE EXISTING CODE
    * Use comments to divide code up, instead of dividing into many functions unless there is severe repetition.
    * Mark documentation comments with a *, i.e. `# * Comment here`
    * Minimal comments in general aside from dividers.
    * Leave comments in place unless you are pretty sure its an AI comment, in which case feel free to remove.
    * Please DO COMMENT ANYTHING REALLY SURPRISING. Use `# *** Comment here` so these stand out.
    * Document key functions but you don't need to document all functions.
    * I like tensor shape comments, especially when its not obvious, but don't spam them.

- Better sorry than safe: 
    * Prefer HARD FAILURE over soft failure, always.
    * NEVER absorb and exception in order to continue, let it raise and crash instead.
    * Never invent dummy data to keep going if data is missing.
    * Don't print warnings if it changes the result, always throw an exception.
    * If something is missing to continue in a code path, HARD FAIL immediately. Don't try to generate dummy data or simplified to continue without solving the hard problem.
    * Don't check if filepaths provided as input exist, you can assume they do and just let it fail if they don't. 
    * The same goes for the innards of input data, e.g. json or tables. You can assume they are well-structured.
    * Assertions are fine especially if they cause early failure, but you don't need to assert something if regular code would fail anyways.
    
- Arg parsing:
    * In Python, parse args with Tyro always
    * Include 1 letter aliases for the key commands only, like: `Annotated[bool, arg(aliases=["-y"])]`

- Double check yourself:
    * Look at the code you wrote and ask yourself if there is any way it could be wrong, or if there is any edge case you forgot to consider. 
    * Make sure to update any other script using the features you changed.

## Codebase overview 
The following is an automatically generated overview of the codebase.

### Project map
- `gray/`: Python orchestration for config, scene loading, training loop, and PyTorch wrapper around the native raytracer.
- `cuda/`: CUDA/OptiX backend (forward/backward rendering, optimizer kernels, BVH/pipeline wrappers, tensor-backed data holders).
- `viewer/` + `view.py`: interactive viewer and websocket plumbing; can run local viewer while training.

### Canonical workflows (use existing files/scripts)
- Activate environment: `source .venv/bin/activate`.
- Compile CUDA code: `bash ./make.sh`.
- One-shot run helper (USE THIS): `bash run.sh <out> -s <scene> -r <level>`.

### Architecture and data flow
- Python loads the native extension via `torch.classes.load_library` and instantiates `torch.classes.quicktracer.Raytracer` (`gray/raytracer.py`, `cuda/raytracer.cpp`).
- Native module exposes grouped tensor holders (`get_config()`, `get_gaussians()`, `get_camera()`, etc.); Python code mutates these tensors in-place instead of copying large state.
- Scene pipeline expects COLMAP-style data and writes intermediate caches/artifacts (`.safetensors`, camera JSON metadata) used by training/rendering/viewing (`gray/scene.py`, `train.py`).
- Keep data CUDA side when possible (e.g. Gaussian data)

### Critical invariants
- Preserve forward/backward contract: backward relies on forward-pass state (camera/framebuffer/related tensors) staying unmodified before `backward()`.
- Keep Python/CUDA API names synchronized when editing bindings (`TORCH_LIBRARY(quicktracer, ...)` surface in `cuda/raytracer.cpp` and calls in `gray/raytracer.py`).
- For train+viewer concurrency, respect existing lock usage around shared raytracer state (`gaussian_lock` pattern in `train.py`/`view.py`).

### Non-standard backward and optimizer flow
- Training uses `raytracer.backward(loss)` then `raytracer.step()` (see `train.py`), not a standard `torch.optim` step on gaussian tensors.
- In `Raytracer.backward`, Python first runs `loss.backward()` for autograd-connected parts (e.g., MLPs), then copies `output_channels.grad` into CUDA framebuffer grad buffers, then calls `self.cuda_module.backward_pass()`.
- Gaussian optimization is CUDA-side: `self.cuda_module.step()` applies updates using Adam moments/lrs stored in tensor holders (`get_gaussians()` state), then `update_bvh()` runs.
- Keep this two-stage contract intact when editing: gradients are bridged from PyTorch to CUDA explicitly, and parameter updates for gaussians happen in native code.

### Project-specific coding conventions
- Use Tyro for Python CLI/dataclass argument parsing (`gray/config.py`, script entrypoints).
- Prefer straightforward top-down control flow; only extract helpers when repetition is substantial.
- Prefer hard failure over fallback behavior; do not fabricate dummy data for missing inputs.
- Keep comments minimal; add comments for genuinely surprising behavior.

### Integration points and dependencies
- Build stack is CMake + Ninja + CUDA + OptiX 8 + Torch (`CMakeLists.txt`, `cmake/FindOptiX8.cmake`, `cmake/FindTorch.cmake`).
- Optional tiny-cuda-nn path is installed via `uv sync --extra tcnn`, YOU DON'T NEED TO INSTALL IT.
- Dataset prep helpers live in `scripts/` plus `convert.py` and `resize.py`; dense init uses `third_party/edgs.py`.
