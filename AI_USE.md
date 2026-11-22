# AI Use Disclosure

DSA 8401 Applied Machine Learning capstone
Student: Linet Wanjira: 193883

## 1. Tools used

- Claude (Anthropic), through the claude.ai chat interface.
- No other AI tool
- Period of use: 17-20 November 2026.

## 2. Summary

I used Claude as a planning assistant, a code collaborator, a debugging helper and a tutor.


## 3. What the AI helped with

| Area | AI contribution | Files |
|---|---|---|
| Planning | Read the brief and my proposal; produced a plan, a risk list (real-world messiness rule, leakage, grouped split) and a rubric-to-work mapping | none |
| Environment | Step-by-step setup of the GitHub repo, venv and dependencies; diagnosed the TensorFlow install failure on Windows (long file paths) | requirements.txt, requirements-api.txt |
| Data generation | Reviewed my pattern-drafting code; found invalid pocket polygons (self-intersections) and proposed the fix; proposed the stratified size plan with within-size variation | src/data/generate_patterns.py |
| Layout and dataset | Wrote the first-fit shelf-packing layout engine, the fixed-scale rasteriser and the dataset builder (noise, missing values, grouped stratified split) | src/data/layout.py, rasterize.py, build_dataset.py |
| Modelling | Wrote preprocessing, the MLP, CNN and fusion models, the training script and the evaluation script with baselines | src/features/preprocess.py, src/models/, src/train.py, src/evaluate.py |
| Interpretation | Suggested the calibrated-formula baseline and helped me understand why fusion reaches R2 0.996 (area from measurements, length from the mask) | reports/ |
| Deployment | Wrote the FastAPI service, smoke tests, example payload script, Dockerfile, docker-compose and the pipeline runner scripts | api/, tests/, scripts/, Dockerfile, docker-compose.yml, run_pipeline.* |
| Documentation | Drafted the README template and this disclosure, I added the industry understanding in the problem statement and challenge definition | README.md, AI_USE.md |
| Study support | Produced Q&A study notes for the presentation | not in the repo |
| Debugging | Helped resolve errors during setup and running (pip and venv, module paths, a NameError after a refactor, test collection) | various |

## 4. What I did myself

- The proposal and the project idea.
- The pattern-drafting functions code and assisted polygon fixing (29th September) using Gemini (front and back panels, waistband, fly, pockets, SVG export). 
- Running every script and test on my own machine.
- Choosing the fabric level and model settings (fabric width 150 cm, 2.1 cm per pixel, 5 layouts per garment).
- Interpreting the results and writing the report and slides (the report and slides are my own writing; the AI gave only pointer notes).

## 5. How I verified AI output

- I ran the scripts locally. The API smoke tests pass (4 passed) and I reproduced the evaluation table on my machine.
- The AI also ran its code against a replica of my geometry before giving it to me, and checked: no overlapping pieces, no lost area, all layouts inside the fabric, and valid polygons.
- I opened the sample SVGs, checked the 1,200-file count and the size counts, reading the code line by line.

## 6. Errors in AI output that were found and corrected

- The first estimate of cm_per_pixel (1.2-1.5) was wrong; after measuring real layout lengths it was corrected to 2.1.
- Invalid (self-intersecting) pocket polygons, about 5% of pieces, were caught by a validity check and fixed.
- A Keras 3 incompatibility in the training code (how single-input models are fed) was found when the code was tested and was fixed.
- Assumed WSL (Window Subsystem Linux) was set-up for Docker.

## 7. Where the project differs from my proposal

See the table in README.md or the report: Shapely and Pillow instead of the TensorFlow geometry and rasterisation tools, 5 layouts per garment (500 samples), additional baselines, and added measurement noise, missing values and a grouped split.

The shapely and pillow libraries provided better polygons for the smaller items. The large objects have little difference though.

## 8. Responsibility

I understand the problem statement, the expected outcome and the goal that needs to be achieved in sustainability. I have a good understanding of how submitted code works. I take responsibility for the submission.

## 9. Record

The full chat history is available on request, though it does contain personally identifiable information.