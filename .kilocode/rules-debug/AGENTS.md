# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Debug Mode Specific Guidelines

### Turkish-Named Classes
Error messages about missing classes will use Turkish keys. Common class names to watch for:
- `manyetit`, `krom`, `atık`, `düşük tenör`, `çatlak`, `çizik`, `delik`, `leke`, `deformasyon`
- When debugging KeyError or AttributeError, check if Turkish dictionary keys are being used

### YOLO Model Loading Failures
Model loading failures are often silent. Debug steps:
1. Check model paths in `models/` directory
2. Verify `models/registry.yaml` contains correct model configurations
3. Run `python models/download_models.py` to ensure models are downloaded
4. Check Ultralytics version compatibility (v8+ required)

### Streamlit Debugging
Streamlit reruns entire script on interaction. Debugging requires:
- Use `st.session_state` to inspect state between reruns
- Add `st.write()` or `st.code()` for debugging output
- Use `st.fragment` decorator for partial reruns (Streamlit 1.35+)
- Check browser console for JavaScript errors

### HuggingFace Space Debugging
The `huggingface_space/` uses Gradio instead of Streamlit:
- Class names differ from main app (`çizik`, `çatlak`, `delik`, `ezilme`, `yanık`, `pas`, `diğer`)
- Check `huggingface_space/requirements.txt` for dependencies
- Use Gradio's debug mode: `python huggingface_space/app.py --debug`
