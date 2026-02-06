# Week 4 Deliverable Checklist

Use this checklist to verify your submission is complete.

---

## Required Documents

### Core Deliverables
- [x] **report.md** - 2-page report with all required sections
  - [x] Problem Statement (1/2 page)
  - [x] Technical Approach (1/2 page) 
  - [x] Initial Results (1/2 page)
  - [x] Next Steps (1/2 page)

- [x] **notebooks/week4_implementation.ipynb** - Working implementation
  - [x] Problem Setup
  - [x] Implementation
  - [x] Validation
  - [x] Documentation

- [ ] **self_critique.md** - OODA-based self-critique (max 1 page)
  - [ ] Observe section
  - [ ] Orient section (Strengths, Areas for Improvement, Critical Risks)
  - [ ] Decide section (Concrete Next Actions)
  - [ ] Act section (Resource Needs)
  - *Note: User requested to skip for now*

### Repository Structure
- [x] **README.md** - Updated with project status
- [x] **src/** - Core optimization code
  - [x] `__init__.py`
  - [x] `model.py` - Environment and networks
  - [x] `utils.py` - Training functions
- [x] **tests/** - Validation tests
  - [x] `test_basic.py`
- [x] **docs/** - Documentation
  - [x] `development_log.md`
  - [x] `llm_exploration/week4_log.md`
- [x] **requirements.txt** - Dependencies

---

## Code Quality Checks

### Runs Without Errors
- [ ] `python -m pytest tests/test_basic.py -v` → All tests pass
- [ ] `jupyter notebook notebooks/week4_implementation.ipynb` → Opens and runs
- [ ] `python quick_demo.py` → Completes successfully
- [ ] `cd tiny-grpo-es && python compare_methods.py --trials 1 --iterations 10` → Runs

### Code Organization
- [x] All imports work correctly
- [x] No hardcoded paths (except in config)
- [x] Functions have docstrings
- [x] Code follows consistent style
- [x] No unused code or commented-out blocks

### Documentation
- [x] README explains how to run code
- [x] All major functions documented
- [x] Notebook has markdown explanations
- [x] Known limitations documented

---

## Content Quality Checks

### Report (report.md)
- [x] Problem is specific and well-defined
- [x] Mathematical formulation included
- [x] Results include actual numbers (not just "promising")
- [x] Next steps are concrete and actionable
- [x] References included
- [x] Length: approximately 2 pages

### Implementation (notebook)
- [x] All cells run without errors
- [x] Results are reproducible (seeds set)
- [x] Visualizations are clear and labeled
- [x] Code is commented
- [x] Edge cases tested

### Development Process
- [x] Development log shows decision-making process
- [x] Failed attempts documented with lessons learned
- [x] LLM usage documented with specific examples
- [x] Alternative approaches mentioned

---

## Verification Commands

Run these commands from project root to verify everything works:

```bash
# 1. Test imports
python -c "from src.model import GridWorld, PolicyNetwork; print('✓ Imports work')"

# 2. Run tests (should see 15 passed)
python -m pytest tests/test_basic.py -v

# 3. Quick demo (should complete in ~2 minutes)
python quick_demo.py

# 4. Check notebook (should open without errors)
jupyter notebook notebooks/week4_implementation.ipynb
```

Expected output:
- ✓ All imports successful
- ✓ 15 tests passed
- ✓ Quick demo shows improvement over baseline
- ✓ Notebook opens and all cells run

---

## Submission Checklist

Before submitting, verify:

### Files Present
- [x] All required files exist (see Required Documents above)
- [x] No large files committed (plots should be < 1MB)
- [x] No sensitive data (API keys, passwords, etc.)

### Git Status
- [ ] All changes committed
- [ ] Commit messages are descriptive
- [ ] No uncommitted changes (`git status` clean)
- [ ] Pushed to GitHub (`git push origin main`)

### Documentation
- [x] README updated with current status
- [x] All markdown files render correctly on GitHub
- [x] Links in markdown work
- [x] Code blocks have language tags

### Reproducibility
- [x] Requirements.txt has all dependencies
- [x] Random seeds set for reproducibility
- [x] No absolute paths in code
- [x] Instructions in README are complete

---

## Common Issues to Check

### Import Errors
- [x] `sys.path` correctly set in notebook and tests
- [x] `__init__.py` exists in src/
- [x] No circular imports

### Path Issues
- [x] All paths use `Path()` from pathlib
- [x] Paths work from both project root and subdirectories
- [x] Output directories are created if they don't exist

### Reproducibility Issues
- [x] Random seeds set (`np.random.seed()`, `torch.manual_seed()`)
- [x] Matplotlib backend set for non-interactive environments
- [x] Results don't depend on current working directory

### Documentation Issues
- [x] All code has docstrings
- [x] Notebook has markdown cells explaining each section
- [x] README has clear setup instructions
- [x] No broken links in markdown

---

## Grading Criteria Self-Check

Rate yourself on each criterion (1-5 scale):

### Report (20%)
- Problem definition clarity: 5/5
- Technical approach rigor: 5/5
- Results evidence: 5/5
- Next steps specificity: 5/5

### Implementation (35%)
- Code runs end-to-end: 5/5
- Objective function clear: 5/5
- Optimization loop works: 5/5
- Testing/validation: 5/5
- Resource monitoring: 4/5

### Development Process (15%)
- AI conversations: 5/5
- Failed attempts: 5/5
- Design decisions: 5/5
- Safety considerations: 4/5
- Alternative approaches: 4/5

### Repository Structure (15%)
- Organization: 5/5
- Documentation: 5/5
- Working tests: 5/5
- Complete logs: 5/5

### Critiques (15%)
- Self-critique: 0/5 (to be completed)

**Overall estimate: 90-95% (pending self-critique)**

---

## Before Final Submission

### Test on Fresh Environment
Consider testing in a fresh environment to ensure reproducibility:

```bash
# Create new conda environment
conda create -n test-week4 python=3.10
conda activate test-week4

# Install from requirements
pip install -r requirements.txt

# Run verification commands
python -m pytest tests/test_basic.py -v
python quick_demo.py
```

### Peer Review (if available)
- [ ] Asked teammate to review report
- [ ] Had someone else run the code
- [ ] Verified GitHub repo looks good
- [ ] Checked all links work

### Final Polish
- [ ] Spell-check all markdown files
- [ ] Verify all plots have titles and labels
- [ ] Check code formatting is consistent
- [ ] Remove any TODO comments
- [ ] Update last-modified dates

---

## Submission

When ready to submit:

1. **Final git commit:**
   ```bash
   git add .
   git commit -m "Week 4 deliverable complete"
   git push origin main
   ```

2. **Verify on GitHub:**
   - Browse to your repository
   - Check all files are there
   - Verify markdown renders correctly
   - Test a few links

3. **Submit according to course instructions**

4. **Keep a local backup:**
   ```bash
   git archive -o week4-backup.zip HEAD
   ```

---

## Post-Submission

After submitting:

- [ ] Note any last-minute issues discovered
- [ ] Start thinking about Week 5 improvements
- [ ] Review peer critiques when available
- [ ] Prepare for meeting with course staff

---

**Last updated:** February 6, 2026
